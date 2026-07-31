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

        # Optional wall-clock breakdown (SGLANG_BEAM_CASCADE_PROFILE=1). Used to
        # attribute the cascade overhead between the per-step CPU plan and the
        # per-layer attention/merge kernels. Off by default (zero overhead).
        from sglang.srt.environ import envs

        self._profile = envs.SGLANG_BEAM_CASCADE_PROFILE.get()
        self._prof = {"plan_s": 0.0, "fwd_s": 0.0, "plan_n": 0, "fwd_n": 0}

        # CUDA-graph mode state (initialized by init_graph_state when
        # SGLANG_BEAM_CASCADE_CAPTURE_BEAM_WIDTH > 0). Per-bucket wrappers are
        # created lazily at capture time.
        self._workspace_buffer = attn_backend.workspace_buffer
        self._graph_K: int = 0
        self._graph_wrappers = {}  # bucket bs -> (shared_wrapper, unique_wrapper)
        self._graph_kv_indices_s: Optional[torch.Tensor] = None
        self._graph_kv_indices_u: Optional[torch.Tensor] = None

        # Per-step path counter, dumped to a fixed file so a bench run can be
        # attributed definitively (eager plan vs graph capture/replay). Near
        # zero cost: two dict increments per decode step + a tiny file write
        # every 16 steps.
        self._mode_counts = {"eager_plan": 0, "graph_capture": 0, "graph_replay": 0}

    def _bump_mode(self, key: str):
        self._mode_counts[key] += 1
        total = sum(self._mode_counts.values())
        if total % 16 == 0:
            try:
                with open("/tmp/sglang_beam_cascade_mode.txt", "w") as f:
                    f.write(
                        f"[beam-cascade mode] {self._mode_counts} "
                        f"(graph_K={self._graph_K}, "
                        f"captured_buckets={sorted(self._graph_wrappers)})\n"
                    )
            except OSError:
                pass

    def profile_summary(self) -> str:
        p = self._prof
        return (
            f"[beam-cascade] plan: {p['plan_s'] * 1e3:.1f}ms over {p['plan_n']} steps "
            f"({p['plan_s'] / max(p['plan_n'], 1) * 1e3:.3f}ms/step) | "
            f"fwd: {p['fwd_s'] * 1e3:.1f}ms over {p['fwd_n']} layer-calls "
            f"({p['fwd_s'] / max(p['fwd_n'], 1) * 1e3:.3f}ms/call)"
        )

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
        if not self._profile:
            return self._plan_impl(forward_batch)

        import time

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        md = self._plan_impl(forward_batch)
        torch.cuda.synchronize()
        self._prof["plan_s"] += time.perf_counter() - t0
        self._prof["plan_n"] += 1
        if self._prof["plan_n"] % 16 == 0:
            # Overwrite a fixed file instead of logging (scheduler logs are
            # noisy/hard to reach in bench runs); `cat` it anytime.
            try:
                with open("/tmp/sglang_beam_cascade_profile.txt", "w") as f:
                    f.write(self.profile_summary() + "\n")
            except OSError:
                pass
        return md

    def _plan_impl(
        self, forward_batch: ForwardBatch
    ) -> Optional[BeamCascadeMetadata]:
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

        self._bump_mode("eager_plan")
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
        if not self._profile:
            return self._forward_impl(q, kv_cache, layer, metadata)

        import time

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = self._forward_impl(q, kv_cache, layer, metadata)
        torch.cuda.synchronize()
        self._prof["fwd_s"] += time.perf_counter() - t0
        self._prof["fwd_n"] += 1
        return o

    def _forward_impl(
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

    # ------------------------------------------------------------------
    # CUDA graph mode
    #
    # Same two-level kernel sequence, but captured in a CUDA graph. Follows
    # the EAGLE draft-extend pattern: wrappers own fixed input buffers
    # (use_cuda_graph=True); the first real begin_forward at capture builds
    # _cached_module, after which begin_forward is swapped for
    # fast_prefill_plan (sync-free, refreshes _plan_info before each replay).
    # Only uniform-beam-width batches whose size exactly matches a captured
    # bucket run this path; everything else falls back to eager cascade.
    # ------------------------------------------------------------------

    # Per-beam decode segment capacity in graph mode. Decode segments only
    # hold generated tokens, so a moderate cap keeps the U-level buffers small;
    # batches exceeding it fall back to eager cascade.
    GRAPH_UNIQUE_SEG_CAP = 2048
    # Shared prompt capacity in graph mode. Sizing the S-level buffers by the
    # model's full context length would waste tens of MB per bucket, so cap it
    # and fall back to eager cascade for longer prompts.
    GRAPH_SHARED_SEG_CAP = 8192
    # Cap the number of captured buckets: each one owns its own fixed index
    # buffers, so an unbounded [K, 2K, 4K, ...] ladder would add up.
    GRAPH_MAX_BUCKETS = 4

    def graph_bucket_list(self, max_bs: int):
        """Cascade capture buckets: power-of-two multiples of K up to max_bs."""
        K = self._graph_K
        if K <= 0:
            return []
        buckets = []
        m = 1
        while m * K <= max_bs and len(buckets) < self.GRAPH_MAX_BUCKETS:
            buckets.append(m * K)
            m *= 2
        return buckets

    def init_graph_state(self, max_bs: int):
        """Allocate graph-mode staging buffers (idempotent)."""
        from sglang.srt.environ import envs

        K = envs.SGLANG_BEAM_CASCADE_CAPTURE_BEAM_WIDTH.get()
        if K <= 0 or self._graph_K == K:
            return
        self._graph_K = K
        buckets = self.graph_bucket_list(max_bs)
        if not buckets:
            self._graph_K = 0
            return
        top_bs = buckets[-1]
        max_reqs = max(top_bs // K, 1)
        # Staging buffers the triton index builder writes into; fast_prefill_plan
        # then copies them into each wrapper's own fixed buffer (kept separate to
        # avoid an aliased self-copy inside fast_prefill_plan).
        self._graph_kv_indices_s = torch.zeros(
            max_reqs * self.GRAPH_SHARED_SEG_CAP, dtype=torch.int32, device=self.device
        )
        self._graph_kv_indices_u = torch.zeros(
            top_bs * self.GRAPH_UNIQUE_SEG_CAP, dtype=torch.int32, device=self.device
        )

    def _ensure_graph_wrappers(self, bs: int):
        """Create the per-bucket wrapper pair with fixed input buffers."""
        if bs in self._graph_wrappers:
            return self._graph_wrappers[bs]

        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        K = self._graph_K
        num_reqs = bs // K
        device = self.device

        def _make(num_rows: int, indices_cap: int):
            return BatchPrefillWithPagedKVCacheWrapper(
                self._workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                # fast_prefill_plan asserts the fa2 backend.
                backend="fa2",
                qo_indptr_buf=torch.zeros(
                    num_rows + 1, dtype=torch.int32, device=device
                ),
                paged_kv_indptr_buf=torch.zeros(
                    num_rows + 1, dtype=torch.int32, device=device
                ),
                paged_kv_indices_buf=torch.zeros(
                    indices_cap, dtype=torch.int32, device=device
                ),
                paged_kv_last_page_len_buf=torch.ones(
                    num_rows, dtype=torch.int32, device=device
                ),
            )

        pair = (
            # Level-S: one row per request, shared prompt segment
            _make(num_reqs, num_reqs * self.GRAPH_SHARED_SEG_CAP),
            # Level-U: one row per beam, own decode segment
            _make(bs, bs * self.GRAPH_UNIQUE_SEG_CAP),
        )
        self._graph_wrappers[bs] = pair
        return pair

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        """Whether this beam batch matches a capturable/captured cascade shape.

        Requires: graph mode initialized, uniform beam width == captured K,
        exact bucket match (no padding in v1), contiguous groups, and decode
        segments within the U-level capacity.
        """
        beam_widths = forward_batch.beam_widths
        if not beam_widths or self._graph_K <= 0:
            return False
        K = self._graph_K
        if any(w != K for w in beam_widths):
            return False
        total = len(beam_widths) * K
        # Only shapes that were captured are replayable (no padding in v1).
        if total not in self._graph_wrappers:
            return False
        expect = 0
        for w, s in zip(beam_widths, forward_batch.beam_slot_starts):
            if s != expect:
                return False
            expect += w
        if forward_batch.seq_lens_cpu is None:
            return False
        # Prompt / decode segments must fit the fixed graph buffers.
        if any(p > self.GRAPH_SHARED_SEG_CAP for p in forward_batch.beam_prompt_lens):
            return False
        for seq, p in zip(
            forward_batch.seq_lens_cpu.tolist(),
            [p for p in forward_batch.beam_prompt_lens for _ in range(K)],
        ):
            if not 0 < int(seq) - p <= self.GRAPH_UNIQUE_SEG_CAP:
                return False
        return True

    def plan_for_graph(
        self, forward_batch, bs: int, in_capture: bool
    ) -> BeamCascadeMetadata:
        """Refresh the graph-mode wrappers for this step.

        in_capture=True: run one real begin_forward on a self-built dummy
        layout (builds _cached_module and records buffer bindings), then swap
        begin_forward for fast_prefill_plan.
        in_capture=False (replay prep): rebuild indices into the staging
        buffers and call fast_prefill_plan with host-known layout, no D2H.
        """
        from functools import partial

        from sglang.srt.layers.attention.flashinfer_backend import fast_prefill_plan

        shared_w, unique_w = self._ensure_graph_wrappers(bs)
        K = self._graph_K
        num_reqs = bs // K
        device = self.device
        common = dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            non_blocking=True,
        )

        if in_capture:
            # Dummy but valid layout: prompt_len=1, unique_len=1 per row. The
            # capture warmup only needs legal shapes (values are discarded);
            # replay-time fast_prefill_plan rewrites the real layout. Uses the
            # original plan() so _cached_module and the fixed-buffer bindings
            # are established (positional args match the eager call style).
            qo_s = torch.arange(0, bs + 1, K, dtype=torch.int32, device=device)
            kv_s = torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
            idx_s = torch.zeros(num_reqs, dtype=torch.int32, device=device)
            shared_w.begin_forward(
                qo_s,
                kv_s,
                idx_s,
                self._ones(num_reqs),
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                non_blocking=True,
            )
            qo_u = torch.arange(bs + 1, dtype=torch.int32, device=device)
            kv_u = torch.arange(bs + 1, dtype=torch.int32, device=device)
            idx_u = torch.zeros(bs, dtype=torch.int32, device=device)
            unique_w.begin_forward(
                qo_u,
                kv_u,
                idx_u,
                self._ones(bs),
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                non_blocking=True,
            )
            # From now on every plan on these wrappers is the sync-free path.
            shared_w.begin_forward = partial(fast_prefill_plan, shared_w)
            unique_w.begin_forward = partial(fast_prefill_plan, unique_w)
            self._bump_mode("graph_capture")
        else:
            self._plan_graph_replay(forward_batch, bs, shared_w, unique_w, common)
            self._bump_mode("graph_replay")

        return BeamCascadeMetadata(
            shared_wrapper=shared_w,
            unique_wrapper=unique_w,
            num_reqs=num_reqs,
            total_beams=bs,
        )

    def _plan_graph_replay(self, forward_batch, bs, shared_w, unique_w, common):
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_flashinfer_kv_indices_triton,
        )

        K = self._graph_K
        num_reqs = bs // K
        device = self.device
        prompt_lens = forward_batch.beam_prompt_lens
        slot_starts = forward_batch.beam_slot_starts
        seq_lens_cpu = forward_batch.seq_lens_cpu.tolist()
        req_pool_indices = forward_batch.req_pool_indices

        # ---- host-side layout (all values CPU-known; no device sync) ----
        qo_s_host = torch.arange(0, bs + 1, K, dtype=torch.int32)
        kv_s_host = torch.zeros(num_reqs + 1, dtype=torch.int32)
        kv_s_host[1:] = torch.cumsum(
            torch.tensor(prompt_lens, dtype=torch.int32), dim=0
        )
        kv_lens_s_host = torch.tensor(prompt_lens, dtype=torch.int32)

        unique_lens = [
            int(s) - p
            for s, p in zip(seq_lens_cpu, (p for p in prompt_lens for _ in range(K)))
        ]
        qo_u_host = torch.arange(bs + 1, dtype=torch.int32)
        kv_u_host = torch.zeros(bs + 1, dtype=torch.int32)
        kv_u_host[1:] = torch.cumsum(
            torch.tensor(unique_lens, dtype=torch.int32), dim=0
        )
        kv_lens_u_host = torch.tensor(unique_lens, dtype=torch.int32)

        # ---- device-side indices into the staging buffers ----
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.int32, device=device)
        kv_indptr_s_dev = kv_s_host.to(device, non_blocking=True)
        rep_rows = req_pool_indices[
            torch.tensor(slot_starts, dtype=torch.int64, device=device)
        ]
        create_flashinfer_kv_indices_triton[(num_reqs,)](
            self.req_to_token,
            rep_rows,
            prompt_lens_t,
            kv_indptr_s_dev,
            None,
            self._graph_kv_indices_s,
            self.req_to_token.shape[1],
        )

        prompt_start_t = torch.tensor(
            [p for p in prompt_lens for _ in range(K)],
            dtype=torch.int32,
            device=device,
        )
        unique_lens_t = torch.tensor(unique_lens, dtype=torch.int32, device=device)
        kv_indptr_u_dev = kv_u_host.to(device, non_blocking=True)
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            unique_lens_t,
            kv_indptr_u_dev,
            prompt_start_t,
            self._graph_kv_indices_u,
            self.req_to_token.shape[1],
        )

        # ---- sync-free plan refresh (fast_prefill_plan installed at capture)
        shared_w.begin_forward(
            qo_s_host.to(device, non_blocking=True),
            kv_indptr_s_dev,
            self._graph_kv_indices_s[: int(kv_s_host[-1])],
            self._ones(num_reqs),
            **common,
            qo_indptr_host=qo_s_host,
            kv_indptr_host=kv_s_host,
            kv_lens_host=kv_lens_s_host,
            max_q_len=K,
            max_kv_len=int(kv_lens_s_host.max()),
        )
        unique_w.begin_forward(
            qo_u_host.to(device, non_blocking=True),
            kv_indptr_u_dev,
            self._graph_kv_indices_u[: int(kv_u_host[-1])],
            self._ones(bs),
            **common,
            qo_indptr_host=qo_u_host,
            kv_indptr_host=kv_u_host,
            kv_lens_host=kv_lens_u_host,
            max_q_len=1,
            max_kv_len=int(kv_lens_u_host.max()),
        )


def beam_cascade_supported(layer) -> bool:
    """Per-layer guard: fall back when the layer needs features the cascade
    path does not handle (fp8 KV scales, cross attention, sliding window)."""
    return (
        layer.k_scale is None
        and layer.v_scale is None
        and not layer.is_cross_attention
        and getattr(layer, "sliding_window_size", -1) in (-1, None)
    )
