import copy
import weakref
from typing import List, Tuple

from altair import Optional
import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import Top1Proposer, EagleTreeProposer
from vllm.spec_decode.multi_step_worker import MultiStepWorker


class EagleWorker(MultiStepWorker):
    """The EagleWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.
    The EagleWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for EagleWorker support.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lazy initialization list.
        self._proposer: Top1Proposer

    def init_device(self):
        super().init_device()

    def load_model(self):
        self.model_runner.load_model()

        if (self.model_runner.model.config.tree_choices == [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]):
            self._proposer = Top1Proposer(
                weakref.proxy(self),  # type: ignore[arg-type]
                self.device,
                self.vocab_size,
                max_proposal_len=self.max_model_len,
            )
        else:
            self._proposer = EagleTreeProposer(
                weakref.proxy(self),  # type: ignore[arg-type]
                self.device,
                self.vocab_size,
                max_proposal_len=self.max_model_len,
            )

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        sample_num: Optional[int] = 1,
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.
        For eagle worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)

        if (self.model_runner.model.config.tree_choices == [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]):
            # Shallow copy input data so modifications (such as appending tokens)
            # do not cause side-effects.
            copied_seq_group_metadata_list = self._shallow_copy_inputs(
                execute_model_req.seq_group_metadata_list)
            copied_execute_model_req = execute_model_req.clone(
                copied_seq_group_metadata_list)

            # Assert enough KV space for sample_len tokens per sequence.
            self._assert_enough_kv_space(execute_model_req.seq_group_metadata_list,
                                        sample_len)

            # Run model sample_len times.
            model_outputs: List[SamplerOutput] = []
            for _ in range(sample_len):
                model_output = super().execute_model(
                    execute_model_req=copied_execute_model_req)
                assert (len(model_output) == 1
                        ), "composing multistep workers not supported"
                model_output = model_output[0]

                self._append_new_tokens(model_output,
                                        copied_seq_group_metadata_list)
                model_outputs.append(model_output)
                copied_execute_model_req.previous_hidden_states.hidden_states = model_output.hidden_states

            return model_outputs, True
        else:
            # Assert enough KV space for sample_len tokens per sequence.
            self._assert_enough_kv_space(execute_model_req.seq_group_metadata_list,
                                        sample_num * sample_len)

            copied_seq_group_metadata_list = self._shallow_copy_inputs(
                execute_model_req.seq_group_metadata_list)
            copied_execute_model_req = execute_model_req.clone(
                copied_seq_group_metadata_list)

            model_output = super().execute_model(
                execute_model_req=copied_execute_model_req)

            # Get sample_num * sample_len Outputs.
            model_outputs_topK = model_output[0][0]
            tree_candidates = model_output[0][1]
            return model_outputs_topK, False, tree_candidates
        
    def defragment_accepted_kv_blocks(self,
                                      execute_model_req: Optional[ExecuteModelRequest] = None,
                                      best_candidate_index: Optional[torch.Tensor] = None,
                                      accepted_token_ids: Optional[torch.Tensor] = None,
                                      previous_hidden_states: Optional[torch.Tensor] = None):
        """Rerun the model forward pass accept_len times."""
        
        # Shallow copy input data so modifications (such as appending tokens)
        # do not cause side-effects.
        copied_seq_group_metadata_list = self._shallow_copy_inputs(
            execute_model_req.seq_group_metadata_list)
        copied_execute_model_req = execute_model_req.clone(
            copied_seq_group_metadata_list)

        num_max_accepted_tokens=max(accepted_token_ids.ne(-1).sum(dim=1).tolist())
        for _ in range(num_max_accepted_tokens):
            for seq_group_metadata, accepted_token in zip(
                    copied_seq_group_metadata_list, accepted_token_ids):
                for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
                    seq = seq_group_metadata.seq_data[seq_id]

                    token_id = accepted_token[_].item()
                    seq.append_token_id(token_id, logprob=0.0)
                    seq.update_num_computed_tokens(1)

            copied_execute_model_req.previous_hidden_states.hidden_states = previous_hidden_states[0][_].unsqueeze(0)

            model_output = super().execute_model(
                    execute_model_req=copied_execute_model_req)
            assert (len(model_output) == 1
                    ), "composing multistep workers not supported"
            model_output = model_output[0]