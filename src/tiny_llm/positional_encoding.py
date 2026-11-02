import torch


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        positions = torch.arange(seq_len)
        num_pairs = dims // 2
        def calculate_rates(i):
            return base ** (-2 * i / dims)
        rates = calculate_rates(torch.arange(num_pairs))
        angles = torch.outer(positions, rates)
        self.num_pairs = num_pairs
        self.traditional = traditional
        self.cos_table = torch.cos(angles)
        self.sin_table = torch.sin(angles)

    def __call__(
        self, x: torch.Tensor, offset: list[slice] | slice | None = None
    ) -> torch.Tensor:
        if offset is None:
            offset = [slice(0, x.shape[1])] * x.shape[0]
        elif isinstance(offset, slice):
            offset = [offset] * x.shape[0]
        cos_table = torch.stack([self.cos_table[s] for s in offset]).unsqueeze(2).to(device=x.device)
        sin_table = torch.stack([self.sin_table[s] for s in offset]).unsqueeze(2).to(device=x.device)
        if self.traditional:
            x_paris = x.reshape(*x.shape[:-1], -1, 2)
            a = x_paris[...,0]
            b = x_paris[...,1]
            result_a = a * cos_table - b * sin_table
            result_b = a * sin_table + b * cos_table
            result = torch.stack((result_a, result_b),-1).flatten(-2)
        else:
            a = x[...,:self.num_pairs]
            b = x[...,self.num_pairs:]
            result_a = a * cos_table - b * sin_table
            result_b = a * sin_table + b * cos_table
            result = torch.cat((result_a, result_b),-1)
        return result.to(dtype=x.dtype)
