"""O(1) lookup for the disjoint visual ranges in vLLM's Triton prefix mask."""

from functools import wraps


def prefix_ends(ranges, width, device):
    import numpy as np
    import torch

    ends = np.full((len(ranges), width), -1, dtype=np.int32)
    for row, spans in zip(ends, ranges, strict=True):
        for start, end in sorted(spans):
            if start == end:
                continue  # vLLM's original mask excludes degenerate ranges.
            if not 0 <= start < end < width or np.any(row[start:end + 1] >= 0):
                raise ValueError(f"Unexpected or overlapping Molmo2 prefix range {(start, end)}")
            row[start:end + 1] = end
    return torch.from_numpy(ends).to(device, non_blocking=True)


def install(context):
    """Keep the existing Triton dot products and the exact prefix mask."""
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops import triton_unified_attention as attention

    original_mask = attention.compute_kv_seq_mask
    old_loop = '''        for i in range(MAX_MM_RANGES):
            range_start = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
            )
            range_end = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
            )
            is_valid = range_start < range_end
            q_in_range = (
                (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
            )
            k_in_range = (
                (seq_offset[None, :] >= range_start)
                & (seq_offset[None, :] <= range_end)
                & is_valid
            )
            mm_mask = q_in_range & k_in_range
            if MM_PREFIX_CLAMP_SW and SLIDING_WINDOW > 0:
                mm_mask = mm_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
            seq_mask |= mm_mask'''
    lookup = '''        if SLIDING_WINDOW > 0 or MM_PREFIX_CLAMP_SW:
            tl.static_assert(False, "This prefix index is only valid without a sliding window")
        end = tl.load(
            mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + query_abs_pos,
            mask=(query_abs_pos >= 0) & (query_abs_pos < seq_len),
            other=-1,
        )
        seq_mask |= (seq_offset[None, :] > query_abs_pos) & (seq_offset[None, :] <= end)'''
    if original_mask.src.count(old_loop) != 1:
        raise RuntimeError("Installed vLLM prefix mask source has changed")
    original_mask._unsafe_update_src(original_mask.src.replace(old_loop, lookup))

    # The launcher uses the second dimension of (B, slots, 2) to calculate
    # storage stride. Store one endpoint per position in that same buffer.
    builder = triton_attn.TritonAttentionMetadataBuilder
    old_build = builder.build
    @wraps(old_build)
    def build(self, *args, **kwargs):
        record = old_build(self, *args, **kwargs)
        ranges = record.mm_prefix_range
        if ranges and record.mm_prefix_range_tensor is not None:
            if not record.causal or len(ranges) != 1:
                raise ValueError("Molmo2 prefix lookup requires one causal request")
            endpoints = prefix_ends([ranges[0]], context, record.seq_lens.device)
            record.mm_prefix_range_tensor = endpoints.view(1, context // 2, 2)
        return record

    builder.build = build
    return original_mask.src, attention.kernel_unified_attention.src


def install_flash_prefill():
    """Use FA2 for a full single-query prefill with the exact visual mask.

    A visual span sees its entire causal prefix up to the end of the span.
    Group a few neighboring spans, give FA2 that union of keys, and combine
    it with the preceding dense prefix using the two log normalizers.
    Decode and partial prefill fall back to the existing paged attention.
    """
    import torch
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    original = TritonAttentionImpl.forward
    span_batch = 12

    def dispatch(q, k, v, q_lengths, k_lengths, scale, causal, cache):
        # Reuse GPU prefix pointers across all decoder layers for this request.
        key = tuple(q_lengths), tuple(k_lengths)
        pointers = cache.get(key)
        if pointers is None:
            def cumulative(lengths):
                ends, total = [0], 0
                for n in lengths:
                    total += n
                    ends.append(total)
                return torch.tensor(ends, dtype=torch.int32, device=q.device)

            pointers = cumulative(q_lengths), cumulative(k_lengths)
            cache[key] = pointers
        return flash_attn_varlen_func(
            q, k, v, max(q_lengths), pointers[0], max(k_lengths),
            cu_seqlens_k=pointers[1], softmax_scale=scale, causal=causal,
            return_softmax_lse=True, fa_version=2,
        )

    @wraps(original)
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if (attn_metadata is None or attn_metadata.max_query_len < 2048
                or attn_metadata.max_query_len != attn_metadata.max_seq_len
                or attn_metadata.num_actual_tokens != attn_metadata.max_query_len
                or not attn_metadata.mm_prefix_range):
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output,
                            output_scale, output_block_scale)
        if (not attn_metadata.causal or set(attn_metadata.mm_prefix_range) != {0}
                or output_scale is not None or output_block_scale is not None
                or tuple(self.sliding_window) != (-1, -1) or self.alibi_slopes is not None
                or self.sinks is not None or self.logits_soft_cap not in (None, 0.0)):
            raise ValueError("Molmo2 FlashAttention expects a single full causal prefill")
        n = attn_metadata.num_actual_tokens
        ranges = [s for s in sorted(attn_metadata.mm_prefix_range[0]) if s[0] < s[1]]
        if any(not 0 <= lo < hi < n for lo, hi in ranges) or any(
            a[1] >= b[0] for a, b in zip(ranges, ranges[1:])
        ):
            raise ValueError("Invalid Molmo2 visual range for full prefill")
        cache = getattr(attn_metadata, "molmo_prefix_pointers", None)
        if cache is None:
            cache = {}
            attn_metadata.molmo_prefix_pointers = cache

        # The baseline supplies valid results for every causal-only query.
        causal_out, _ = dispatch(query[:n], key[:n], value[:n], [n], [n], self.scale, True, cache)
        output[:n].copy_(causal_out)
        for first in range(0, len(ranges), span_batch):
            block = ranges[first:first + span_batch]
            before = block[0][0]
            q_lengths = [hi - lo + 1 for lo, hi in block]
            block_queries = torch.cat([query[lo:hi + 1] for lo, hi in block], dim=0)
            local_lengths = [hi + 1 - before for _, hi in block]
            local_out, local_lse = dispatch(
                block_queries, torch.cat([key[before:hi + 1] for _, hi in block]),
                torch.cat([value[before:hi + 1] for _, hi in block]),
                q_lengths, local_lengths, self.scale, False, cache,
            )
            if before:
                common_out, common_lse = dispatch(
                    block_queries, key[:before], value[:before], [sum(q_lengths)], [before],
                    self.scale, False, cache,
                )
                normalizer = torch.logaddexp(local_lse, common_lse)
                w_local = torch.exp(local_lse - normalizer).transpose(0, 1).unsqueeze(-1)
                w_common = torch.exp(common_lse - normalizer).transpose(0, 1).unsqueeze(-1)
                local_out = (local_out.float() * w_local + common_out.float() * w_common).to(query.dtype)
            for (lo, hi), piece in zip(block, local_out.split(q_lengths), strict=True):
                output[lo:hi + 1].copy_(piece)
        return output

    TritonAttentionImpl.forward = forward
