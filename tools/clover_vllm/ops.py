"""Clover block-diffusion ops, vendored from the Clover-35B-A3B-Preview reference (Apache-2.0, DiffuSpace / Qwen Team).

Original: modeling_clover_moe.py. bd_gdn_* operate on state pools indexed by cache_indices, which is the layout of vLLM's
GDN conv_state (dim, width-1) and ssm_state [H, V, K], so the same functions serve the vLLM preview path unchanged."""
import torch
import torch.nn.functional as F

def bd_gdn_conv1d(
    mixed_qkv: torch.Tensor,
    *,
    clean_conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
    conv_weights: torch.Tensor,
    block_size: int,
    active_valid_mask: torch.Tensor,
    conv_right_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply Clover independent right convolution and SiLU without updating cache.

    ``clean_conv_states`` contains the last ``kernel_size - 1`` committed QKV
    projections. Preview tokens are appended only in a temporary tensor.
    """
    num_tokens, channels = mixed_qkv.shape
    batch_size = cache_indices.numel()
    if num_tokens != batch_size * block_size:
        raise ValueError(
            "BD-GDN preview requires one fixed-size block per request: "
            f"{num_tokens=} != {batch_size=} * {block_size=}"
        )
    if active_valid_mask.numel() != num_tokens:
        raise ValueError(f"{active_valid_mask.numel()=} does not match {num_tokens=}")

    safe_cache_indices = cache_indices.to(torch.long).clamp_min(0)
    history = clean_conv_states.index_select(0, safe_cache_indices)
    if history.shape[1] != channels:
        raise ValueError(
            f"BD-GDN conv cache has {history.shape[1]} channels, expected {channels}"
        )

    valid = (
        active_valid_mask.reshape(batch_size, block_size)
        & (cache_indices >= 0)[:, None]
    )
    active_requests = valid.any(dim=1)
    history = history * active_requests[:, None, None].to(history.dtype)
    current = mixed_qkv.reshape(batch_size, block_size, channels)
    current = current * valid[:, :, None].to(current.dtype)
    convolution_input = torch.cat(
        (history, current.transpose(1, 2)),
        dim=-1,
    )

    weight = conv_weights.reshape(channels, 1, -1)
    output = F.conv1d(
        convolution_input.float(),
        weight.float(),
        groups=channels,
    )
    width = weight.shape[-1]
    expected_shape = (channels, 1, width - 1)
    if tuple(conv_right_weight.shape) != expected_shape:
        raise ValueError(f"Clover right weight shape must be {expected_shape}")
    for distance in range(1, min(width, block_size)):
        output[:, :, :-distance] += (
            current[:, distance:, :].transpose(1, 2).float()
            * conv_right_weight[:, 0, distance - 1].float()[None, :, None]
        )
    output = F.silu(output)

    output = output.transpose(1, 2)
    output = output * valid[:, :, None].to(output.dtype)
    return output.reshape(num_tokens, channels).to(mixed_qkv.dtype)


def bd_gdn_local_memory(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    *,
    clean_ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    active_valid_mask: torch.Tensor,
    scale: float,
    softmax_logit_scale: torch.Tensor,
    softmax_history_bias: torch.Tensor,
    softmax_history_c: torch.Tensor,
) -> torch.Tensor:
    """Clover bidirectional softmax preview with a read-only clean-state bucket.

    Normalized q/k determine block attention logits without beta. The history
    bucket logit is bias + history_c * log(valid_block_size / 32), and its
    value is the gate-decayed recurrent readout from clean [B, H, V, K] state.
    """
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(f"Expected query shape [1, T, H, K], got {query.shape}")
    num_tokens = query.shape[1]
    batch_size = cache_indices.numel()
    if num_tokens % batch_size != 0:
        raise ValueError(
            f"BD-GDN token count {num_tokens} is not divisible by batch {batch_size}"
        )
    block_size = num_tokens // batch_size
    if active_valid_mask.numel() != num_tokens:
        raise ValueError(f"{active_valid_mask.numel()=} does not match {num_tokens=}")

    num_value_heads = value.shape[2]
    if num_value_heads % query.shape[2] or num_value_heads % key.shape[2]:
        raise ValueError("BD-GDN value heads must be divisible by query and key heads")
    query = query.repeat_interleave(num_value_heads // query.shape[2], dim=2)
    key = key.repeat_interleave(num_value_heads // key.shape[2], dim=2)

    valid = active_valid_mask.reshape(batch_size, block_size)
    valid_float = valid[:, :, None].float()
    query_float = query.reshape(
        batch_size, block_size, num_value_heads, query.shape[-1]
    ).float()
    key_float = key.reshape(
        batch_size, block_size, num_value_heads, key.shape[-1]
    ).float()
    value_float = value.reshape(
        batch_size, block_size, num_value_heads, value.shape[-1]
    ).float()
    query_float = query_float * torch.rsqrt(
        query_float.square().sum(dim=-1, keepdim=True) + 1e-6
    )
    key_float = key_float * torch.rsqrt(
        key_float.square().sum(dim=-1, keepdim=True) + 1e-6
    )

    safe_cache_indices = cache_indices.to(torch.long).clamp_min(0)
    clean_state = clean_ssm_states.index_select(0, safe_cache_indices).float()
    active_requests = valid.any(dim=1)
    clean_state = clean_state * active_requests[:, None, None, None].float()

    scaled_query = query_float * scale
    clean_output = torch.einsum(
        "blhk,bhvk->blhv",
        scaled_query,
        clean_state,
    )
    decay = g.reshape(batch_size, block_size, num_value_heads).float().exp()
    history = clean_output * decay[..., None]

    for name, scalar in (
        ("logit_scale", softmax_logit_scale),
        ("history_bias", softmax_history_bias),
        ("history_c", softmax_history_c),
    ):
        if scalar.numel() != num_value_heads:
            raise ValueError(f"Clover {name} must have one entry per value head")
    logit_scale = softmax_logit_scale.reshape(-1).float().to(query_float.device)
    history_bias = softmax_history_bias.reshape(-1).float().to(query_float.device)
    history_c = softmax_history_c.reshape(-1).float().to(query_float.device)
    # [B, H, L(query), S(key)] logits over unscaled l2-normalized q/k.
    logits = torch.einsum("blhk,bshk->bhls", query_float, key_float)
    logits = logits * logit_scale.view(1, -1, 1, 1)
    logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
    history_logit = history_bias.view(1, -1, 1, 1).expand(batch_size, -1, block_size, 1)
    valid_block_size = valid.sum(dim=-1).clamp_min(1).float()
    block_size_log_ratio = torch.log(valid_block_size / 32.0)
    history_logit = history_logit + (
        history_c.view(1, -1, 1, 1) * block_size_log_ratio.view(batch_size, 1, 1, 1)
    )
    weights = torch.softmax(torch.cat((logits, history_logit), dim=-1), dim=-1)
    local_output = torch.einsum(
        "bhls,bshv->blhv",
        weights[..., :-1],
        value_float * valid_float[..., None],
    )
    history_weight = weights[..., -1].permute(0, 2, 1)
    output = history_weight[..., None] * history + local_output
    output = output * valid_float[..., None]
    return output.to(query.dtype).reshape(
        1, num_tokens, num_value_heads, value.shape[-1]
    )


def coreset_torch(logits, valid, top_k, top_p):
    b, t, e = logits.shape
    ids = logits.float().argsort(dim=-1, descending=True, stable=True)[..., :top_k]
    scores = logits.float().softmax(-1).gather(-1, ids) * valid[..., None]
    totals = logits.new_zeros((b, e), dtype=torch.float32)
    totals.scatter_add_(1, ids.reshape(b, -1), scores.reshape(b, -1))
    probs = totals / totals.sum(-1, keepdim=True).clamp_min(1e-9)
    order = probs.argsort(dim=-1, descending=True, stable=True)
    sorted_probs = probs.gather(-1, order)
    keep = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
    keep[:, 0] = True
    allowed = torch.zeros_like(keep).scatter(1, order, keep)
    return logits.masked_fill(valid[..., None] & ~allowed[:, None, :], -float("inf"))
