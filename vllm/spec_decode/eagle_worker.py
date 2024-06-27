import copy
import weakref
from typing import List, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import Top1Proposer
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

        self._proposer = Top1Proposer(
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
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.
        For eagle worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)

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
            execute_model_req.previous_hidden_states.hidden_states = model_output.hidden_states

        return model_outputs, True