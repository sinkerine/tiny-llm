from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from .attention import causal_mask
import torch

if TYPE_CHECKING:
    from .paged_kv_cache import PagedKvMetadata


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
        """

    def release(self):
        """
        Release all resources owned by this cache.

        Request-scoped caches use this when generation finishes or a batch slot
        is removed. Dense caches do not own shared resources, while paged caches
        return their physical pages to a shared pool.
        """
        return None

    def materialize(self):
        """Evaluate owned K/V storage without changing its logical layout."""
        return None

    def update_and_fetch_paged(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> "PagedKvMetadata":
        """
        Update this cache and return paged attention metadata.

        Week 3 caches override this. Dense caches intentionally do not provide
        paged metadata, so calling this on them is a programming error.
        """
        raise NotImplementedError("This KV cache does not support paged attention")

    def rewind(self, n: int):
        """
        Remove the newest n logical tokens from this cache.

        This is needed by speculative decoding when some draft tokens are
        rejected after their K/V has already been written. Implementations may
        drop dense suffixes or return whole pages to a page pool.
        """
        raise NotImplementedError("This KV cache does not support rewind")


class TinyKvFullCache(TinyKvCache):
    def __init__(self, capacity: int | None = None):
        if capacity is not None and (
            not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0
        ):
            raise ValueError("capacity must be a non-negative integer or None")
        self.key_values = None
        self.offset = 0
        self.capacity = capacity
        self.logical_copy_bytes = 0
        self.physical_growth_copy_bytes = 0
        self.slice_write_bytes = 0
        self.growth_copy_bytes = 0

    @property
    def uses_capacity(self) -> bool:
        return self.capacity is not None

    def _logical_key_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self.key_values
        return keys[:, :, : self.offset], values[:, :, : self.offset]

    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        assert key.shape == value.shape
        B, H, S, D = key.shape
        if self.uses_capacity:
            end = self.offset + S
            if end > self.capacity:
                raise ValueError(
                    f"KV cache capacity {self.capacity} exceeded by append ending at {end}"
                )
            if self.key_values is None:
                assert self.offset == 0
                keys = torch.zeros(
                    (B, H, self.capacity, D), dtype=key.dtype, device=key.device
                )
                values = torch.zeros(
                    (B, H, self.capacity, D), dtype=value.dtype, device=value.device
                )
                self.key_values = (keys, values)
            else:
                keys, values = self.key_values
                assert keys.shape == (B, H, self.capacity, D)
                assert values.shape == (B, H, self.capacity, D)
                assert keys.dtype == key.dtype
                assert values.dtype == value.dtype
            if S:
                keys, values = self.key_values
                keys[:, :, self.offset : end].copy_(key)
                values[:, :, self.offset : end].copy_(value)
                self.key_values = (keys, values)
                self.slice_write_bytes += key.nbytes + value.nbytes
            self.offset = end
            logical_keys, logical_values = self._logical_key_values()
            return logical_keys, logical_values, self.offset, mask

        if self.key_values is None:
            assert self.offset == 0
            self.key_values = (key, value)
            self.offset = S
            return key, value, self.offset, mask
        else:
            prev_keys, prev_values = self.key_values
            assert prev_keys.shape == (B, H, self.offset, D)
            assert prev_values.shape == (B, H, self.offset, D)
            copied_bytes = prev_keys.nbytes + prev_values.nbytes
            self.logical_copy_bytes += copied_bytes
            self.physical_growth_copy_bytes += copied_bytes
            self.growth_copy_bytes += copied_bytes
            new_keys = torch.cat([prev_keys, key], dim=2)
            new_values = torch.cat([prev_values, value], dim=2)
            self.key_values = (new_keys, new_values)
            self.offset += S
            return new_keys, new_values, self.offset, mask

    def materialize(self):
        if self.key_values is not None:
            if self.key_values[0].is_cuda:
                torch.cuda.synchronize(self.key_values[0].device)

    def reset(self):
        """Reset the logical request while retaining bounded physical storage."""
        self.offset = 0
        if not self.uses_capacity:
            self.key_values = None

    def rewind(self, n: int):
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= self.offset:
            raise ValueError("rewind length must be between zero and the cache length")
        if n == 0:
            return
        self.offset -= n
        if self.offset == 0:
            if not self.uses_capacity:
                self.key_values = None
            return
        if self.uses_capacity:
            return
        self.key_values = (
            self.key_values[0][:, :, : self.offset],
            self.key_values[1][:, :, : self.offset],
        )
