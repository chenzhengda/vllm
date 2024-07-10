from itertools import chain, count
from typing import Iterator, List, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput, SequenceData,
                           SequenceGroupMetadata,)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.util import (sampler_output_to_torch,
                                   split_batch_by_proposal_len)
from vllm.worker.worker_base import WorkerBase
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer

SeqId = int
TargetSeqId = int
TokenId = int


class EagleScorer(BatchExpansionTop1Scorer):
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_lens_list = proposals.proposal_lens.tolist()
        proposal_token_ids_list = proposals.proposal_token_ids.tolist()

        # Filter the list to ignore -1 proposals.
        proposal_token_ids_list_without_skips = [
            proposals for proposals in proposal_token_ids_list
            if -1 not in proposals
        ]

        proposal_tree_token_ids_list = proposals.tree_candidates.tolist()

        (spec_indices, non_spec_indices,
         target_seq_group_metadata_list) = self._prepare_tree_inputs(
             seq_group_metadata_list=execute_model_req.seq_group_metadata_list,
             proposal_token_ids_list=proposal_tree_token_ids_list,
             proposal_lens_list=proposal_lens_list,
         )

        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=execute_model_req.clone(
                seq_group_metadata_list=target_seq_group_metadata_list, ))
        assert len(target_sampler_output) == 1, "expected single-step output"
        all_tokens = target_sampler_output[0][0]
        all_probs = target_sampler_output[0][1]
        all_scorer_hidden_states = target_sampler_output[0][2]

        return SpeculativeScores(
            probs=all_probs,
            token_ids=all_tokens,
            logprobs=torch.rand(all_tokens.shape[0],
                                all_tokens.shape[1],
                                all_tokens.shape[2],
                                self._scorer_worker.model_config.hf_config.vocab_size,
                                device=all_tokens.device,
                                dtype=torch.float32),
            hidden_states=all_scorer_hidden_states,
        )

    def _create_single_target_seq_group_metadata_of_eagle(
        self,
        seq_group_metadata: SequenceGroupMetadata,
        seq_id: SeqId,
        token_ids: List[TokenId],
    ) -> SequenceGroupMetadata:
        """Create a single target SequenceGroupMetadata.

        Args:
            seq_group_metadata: The metadata for the input sequence.
            seq_id: The input sequence ID.
            # target_seq_id: The corresponding target sequence ID.
            token_ids: The list of token ids that are to be appended to the
                input sequence.
        """
        seq_data = seq_group_metadata.seq_data[seq_id]
        prompt_token_ids = seq_data.get_prompt_token_ids()
        new_output_token_ids = [*seq_data.get_output_token_ids(), *token_ids]

        new_seq_data_dict = {
            seq_id:
            SequenceData(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=new_output_token_ids,
            ),
        }
        # This is a hack. Technically, spec decoding should compute
        # num_lookahead slots at one shot, but instead, it expands the batch
        # and evaluate one by one right now. context_len is seq_len - 1 because
        # the kv cache is filled by a previous batch in the batch expansion.
        for data in new_seq_data_dict.values():
            data.update_num_computed_tokens(data.get_len() - len(token_ids) -
                                            1)

        return SequenceGroupMetadata(
            request_id=seq_group_metadata.request_id,
            is_prompt=seq_group_metadata.is_prompt,
            seq_data=new_seq_data_dict,
            sampling_params=seq_group_metadata.sampling_params,
            block_tables={
                seq_id: seq_group_metadata.block_tables[seq_id],
            },
            lora_request=None,
            token_chunk_size=len(token_ids) + 1,
        )

    def _create_target_seq_group_metadata_of_eagle(
        self,
        input_seq_group_metadata: SequenceGroupMetadata,
        proposal_token_ids: List[List[TokenId]],  # shape: [batch_size, k]
        batch_index: int,
    ) -> List[SequenceGroupMetadata]:
        assert not input_seq_group_metadata.is_prompt, (
            "Speculating on "
            "prompts not yet supported")
        assert len(input_seq_group_metadata.seq_data) == 1, (
            "Beam search "
            "not supported in speculative decoding")
        input_seq_id = next(iter(input_seq_group_metadata.seq_data.keys()))

        target_seq_group_metadata_list: List[SequenceGroupMetadata] = []

        token_ids_to_score = self._get_token_ids_to_score(
            proposal_token_ids[batch_index])

        target_seq_group_metadata = self._create_single_target_seq_group_metadata_of_eagle(
            input_seq_group_metadata,
            input_seq_id,
            proposal_token_ids[batch_index],
        )
        # if len(proposal_token_ids) > 1:
        #     target_seq_group_metadata.num_lookahead_slot_mapping_dirty_offset = (
        #         candidate_index * len(proposal_token_ids[0][0]))
        target_seq_group_metadata_list.append(target_seq_group_metadata)

        return target_seq_group_metadata_list


    def _create_scoring_model_input_of_eagle(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_token_ids: List[List[
            List[TokenId]]],  # shape: [batch_size, num_candidate_seqs, k]
    ) -> List[SequenceGroupMetadata]:
        """Given the original input sequences and proposed tokens from the draft
        model, create a list of target sequences that can be used for scoring.

        target_seq_ids_iter provides sequence ids for the expanded batch,
        fulfilling the requirement that no seq id in the expanded batch is equal
        to the seq id in the original batch.
        """

        if not seq_group_metadata_list:
            return []

        target_seq_group_metadata = list(
            chain.from_iterable(
                self._create_target_seq_group_metadata_of_eagle(
                    seq_group_metadata,
                    proposal_token_ids,
                    i,
                ) for i, seq_group_metadata in enumerate(
                    seq_group_metadata_list)))

        return target_seq_group_metadata


    def _prepare_tree_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_token_ids_list: List[List[List[TokenId]]],
        proposal_lens_list: List[int],
    ) -> Tuple[List[int], List[int], List[SequenceGroupMetadata], int]:
        """Given the input sequences and potentially multiple corresponding
        proposal tokens, create a new batch where each sequence has a single
        query token.
        """

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        spec_seqs, spec_indices = split_batch_by_proposal_len(
            seq_group_metadata_list,
            proposal_lens_list,
            select_proposal_len_zero=False)
        non_spec_seqs, non_spec_indices = split_batch_by_proposal_len(
            seq_group_metadata_list,
            proposal_lens_list,
            select_proposal_len_zero=True)

        target_seq_group_metadata_list = self._create_scoring_model_input_of_eagle(
            seq_group_metadata_list=spec_seqs,
            proposal_token_ids=proposal_token_ids_list)

        num_scoring_tokens = len(target_seq_group_metadata_list)
        target_seq_group_metadata_list.extend(non_spec_seqs)

        return (spec_indices, non_spec_indices, target_seq_group_metadata_list)
