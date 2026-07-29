"""Cascade attention for beam search decode on the FlashInfer backend.

Motivation
----------
During beam-search decode, the K beams of one request share the prompt-prefix
KV slots (storage is deduplicated via req_to_token index copying), but the
regular decode kernel still *reads* the shared prompt KV once per beam, i.e.
K times from HBM. This module implements a two-level cascade so the shared
prompt KV is read only once per request:

- Level-S (shared): for each request, its K beam queries attend to the
  request's prompt KV segment in a single non-causal paged-prefill call
  (one HBM pass over the prompt KV, reused by all K queries).
- Level-U (unique): each beam's query attends to its own decode segment
  ``[prompt_len, seq_len)`` (qo_len == 1 paged-prefill call).
- The two partial results are combined with ``flashinfer.cascade.merge_state``
  (online-softmax state merge; mathematically identical to one full-length
  attention).

Enabled via ``SGLANG_BEAM_SEARCH_CASCADE_ATTN=1`` together with
``--enable-beam-search``; every unsupported condition falls back to the
regular decode path, so correctness never depends on this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass
class BeamCascadeMetadata:
    """Planned per-step state; wrappers carry their own plan internally."""

    shared_wrapper: object  # BatchPrefillWithPagedKVCacheWrapper
    unique_wrapper: object  # BatchPrefillWithPagedKVCacheWrapper
    num_reqs: int
    total_beams: int


class BeamCascadeDecodeHelper:
    """Owns the two cascade wrappers and (re)plans them every decode step."""

    def __init__(self, model_runner, attn_backend):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        from sglang.srt.runtime_context import get_parallel

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads
            // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = attn_backend.flashinfer_kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.device = model_runner.device

        self.shared_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            attn_backend.workspace_buffer, "NHD"
        )
        self.unique_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            attn_backend.workspace_buffer, "NHD"
        )
        # page_size == 1 is enforced by --enable-beam-search, so
        # kv_last_page_len is always ones.
        self._ones_buf = torch.ones(256, dtype=torch.int32, device=self.device)

    def _ones(self, n: int) -> torch.Tensor:
        if n > self._ones_buf.numel():
            self._ones_buf = torch.ones(
                max(n, 2 * self._ones_buf.numel()),
                dtype=torch.int32,
                device=self.device,
            )
        return self._ones_buf[:n]

    def plan(self, forward_batch: ForwardBatch) -> Optional[BeamCascadeMetadata]:
        """Build cascade metadata for this decode step; None means fallback."""
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_flashinfer_kv_indices_triton,
        )

        beam_widths = forward_batch.beam_widths
        prompt_lens = forward_batch.beam_prompt_lens
        slot_starts = forward_batch.beam_slot_starts
        if not beam_widths:
            return None

        num_reqs = len(beam_widths)
        total_beams = sum(beam_widths)
        if forward_batch.req_pool_indices.shape[0] != total_beams:
            # Row layout does not match the beam grouping (e.g. mixed batch).
            return None
        # Beam groups must be contiguous and ordered so that the query rows of
        # request i are exactly [qo_indptr[i], qo_indptr[i+1]).
        expect = 0
        for w, s in zip(beam_widths, slot_starts):
            if s != expect:
                return None
            expect += w

        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            return None
        seq_lens_cpu = seq_lens_cpu.tolist()

        device = self.device
        req_pool_indices = forward_batch.req_pool_indices

        # ---------------- Level-S: shared prompt prefix, one row per request
        qo_indptr_s = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        qo_indptr_s[1:] = torch.cumsum(
            torch.tensor(beam_widths, dtype=torch.int32, device=device), dim=0
        )
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.int32, device=device)
        kv_indptr_s = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        kv_indptr_s[1:] = torch.cumsum(prompt_lens_t, dim=0)
        # Representative row per request: the first beam row of the group. Its
        # [0:prompt_len) segment is identical across the K beams by construction.
        rep_rows = req_pool_indices[
            torch.tensor(slot_starts, dtype=torch.int64, device=device)
        ]
        kv_indices_s = torch.empty(
            int(sum(prompt_lens)), dtype=torch.int32, device=device
        )
        create_flashinfer_kv_indices_triton[(num_reqs,)](
            self.req_to_token,
            rep_rows,
            prompt_lens_t,
            kv_indptr_s,
            None,
            kv_indices_s,
            self.req_to_token.shape[1],
        )
        self.shared_wrapper.begin_forward(
            qo_indptr_s,
            kv_indptr_s,
            kv_indices_s,
            self._ones(num_reqs),
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            non_blocking=True,
        )

        # ---------------- Level-U: per-beam decode segment [prompt_len, seq_len)
        prompt_len_per_beam = []
        for w, p in zip(beam_widths, prompt_lens):
            prompt_len_per_beam.extend([p] * w)
        unique_lens_cpu = [
            int(s) - p for s, p in zip(seq_lens_cpu, prompt_len_per_beam)
        ]
        if any(l <= 0 for l in unique_lens_cpu):
            # seq_len == prompt_len should not happen in beam decode (the first
            # generated token is already appended); be defensive anyway.
            return None
        prompt_start_t = torch.tensor(
            prompt_len_per_beam, dtype=torch.int32, device=device
        )
        unique_lens_t = torch.tensor(unique_lens_cpu, dtype=torch.int32, device=device)
        qo_indptr_u = torch.arange(
            0, total_beams + 1, dtype=torch.int32, device=device
        )
        kv_indptr_u = torch.zeros(total_beams + 1, dtype=torch.int32, device=device)
        kv_indptr_u[1:] = torch.cumsum(unique_lens_t, dim=0)
        kv_indices_u = torch.empty(
            int(sum(unique_lens_cpu)), dtype=torch.int32, device=device
        )
        create_flashinfer_kv_indices_triton[(total_beams,)](
            self.req_to_token,
            req_pool_indices,
            unique_lens_t,
            kv_indptr_u,
            prompt_start_t,
            kv_indices_u,
            self.req_to_token.shape[1],
        )
        self.unique_wrapper.begin_forward(
            qo_indptr_u,
            kv_indptr_u,
            kv_indices_u,
            self._ones(total_beams),
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            non_blocking=True,
        )

        return BeamCascadeMetadata(
            shared_wrapper=self.shared_wrapper,
            unique_wrapper=self.unique_wrapper,
            num_reqs=num_reqs,
            total_beams=total_beams,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        layer,
        metadata: BeamCascadeMetadata,
    ) -> torch.Tensor:
        from flashinfer.cascade import merge_state

        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        # Level-S: beams attend to the shared prompt KV (read once per request).
        o_s, lse_s = metadata.shared_wrapper.forward_return_lse(
            q,
            kv_cache,
            causal=False,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )
        # Level-U: each beam attends to its own decode segment (includes the
        # current token; the query sits at the last position so non-causal
        # attention over the segment is exact).
        o_u, lse_u = metadata.unique_wrapper.forward_return_lse(
            q,
            kv_cache,
            causal=False,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )
        o, _ = merge_state(o_s, lse_s, o_u, lse_u)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


def beam_cascade_supported(layer) -> bool:
    """Per-layer guard: fall back when the layer needs features the cascade
    path does not handle (fp8 KV scales, cross attention, sliding window)."""
    return (
        layer.k_scale is None
        and layer.v_scale is None
        and not layer.is_cross_attention
        and getattr(layer, "sliding_window_size", -1) in (-1, None)
    )
