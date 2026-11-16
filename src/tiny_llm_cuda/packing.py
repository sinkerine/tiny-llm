"""Supplied Hugging Face to group-affine W4 checkpoint packing."""

import torch


def pack_weights(weight, group_size=128, bits=4):
    if bits != 4 or weight.ndim != 2 or weight.shape[-1] % group_size or group_size % 8:
        raise ValueError(
            "W4 needs [N,K], K divisible by group_size, group_size divisible by 8"
        )
    groups = weight.detach().float().reshape(weight.shape[0], -1, group_size)
    lo, hi = groups.amin(-1), groups.amax(-1)
    scales = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
    biases = lo.to(torch.bfloat16)
    q = (
        ((groups - biases.float()[..., None]) / scales.float()[..., None])
        .round()
        .clamp(0, 15)
        .long()
    )
    q = q.reshape(weight.shape[0], -1, 8)
    shifts = torch.arange(0, 32, 4, device=weight.device)
    packed = (q << shifts).sum(-1).to(torch.int32)
    return packed.contiguous(), scales.contiguous(), biases.contiguous()
