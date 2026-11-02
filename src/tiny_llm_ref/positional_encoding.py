import torch


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        half_dims = dims // 2
        inner = torch.arange(0, half_dims, dtype=torch.float32) / half_dims
        freqs = base ** (-inner)
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs)
        self.cos_freqs = torch.cos(freqs)
        self.sin_freqs = torch.sin(freqs)
        self.base = base
        self.half_dims = half_dims
        self.traditional = traditional

    def __call__(
        self, x: torch.Tensor, offset: list[slice] | slice | None = None
    ) -> torch.Tensor:
        N, S, H, D = x.shape
        if offset is not None:
            if isinstance(offset, slice):
                assert offset.stop - offset.start == S, f"offset must be of length {S}"
            elif isinstance(offset, list):
                assert len(offset) == N, (
                    f"offsets must have the same length as batch size {N}"
                )
                for o in offset:
                    assert o.stop - o.start == S, f"offset must be of length {S}"
                offset = torch.tensor([list(range(i.start, i.stop)) for i in offset])
        self.cos_freqs = self.cos_freqs.to(x.device)
        self.sin_freqs = self.sin_freqs.to(x.device)
        cos_basis = (
            self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :]
        )
        sin_basis = (
            self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :]
        )
        # reshape x: (b, s, n_heads, head_dim // 2, 2)
        if self.traditional:
            x = x.reshape(N, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]
        # reshape basis: (1, s, 1, dims // 2, 2)
        cos_basis = cos_basis.reshape(-1, S, 1, self.half_dims)
        sin_basis = sin_basis.reshape(-1, S, 1, self.half_dims)
        # manually doing complex number multiplication..
        real = torch.mul(x1, cos_basis) - torch.mul(x2, sin_basis)
        imag = torch.mul(x2, cos_basis) + torch.mul(x1, sin_basis)
        if self.traditional:
            y = torch.stack([real, imag], dim=-1)
            y = y.reshape(N, S, H, D)
        else:
            y = torch.cat([real, imag], dim=-1)
            y = y.reshape(N, S, H, D)
        return y.to(x.dtype)
