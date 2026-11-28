from dataclasses import dataclass
from typing import Optional

import torch

from . import cuda_ops
from .kv_cache import TinyKvCache


@dataclass
class PagedKvMetadata:
    key_pages: torch.Tensor
    value_pages: torch.Tensor
    block_table: torch.Tensor
    context_lens: torch.Tensor
    page_size: int
    mask: torch.Tensor | str | None = None


class TinyKvPagedPool:
    """Layer-local physical storage for paged KV.

    The model owns one pool per transformer layer. Request caches for the same
    layer share that pool, so a physical page id only needs to be unique within
    its layer.
    """

    def __init__(self, page_size: int = 128):
        assert page_size > 0
        self.page_size = page_size
        self._key_pages: torch.Tensor | None = None
        self._value_pages: torch.Tensor | None = None
        self.free_page_ids: list[int] = []
        self.used_page_ids: set[int] = set()
        self.num_allocated_pages = 0
        self.reused_page_allocations = 0
        self.storage_growths = 0
        self.copied_pages_on_growth = 0
        self.copied_bytes_on_growth = 0

    @property
    def key_pages(self) -> torch.Tensor | None:
        if self._key_pages is None:
            return None
        return self._key_pages[: self.num_pages]

    @property
    def value_pages(self) -> torch.Tensor | None:
        if self._value_pages is None:
            return None
        return self._value_pages[: self.num_pages]

    @property
    def capacity(self) -> int:
        if self._key_pages is None:
            return 0
        return self._key_pages.shape[0]

    @property
    def num_pages(self) -> int:
        return self.num_allocated_pages

    @property
    def num_free_pages(self) -> int:
        return len(self.free_page_ids)

    @property
    def storage_nbytes(self) -> int:
        if self._key_pages is None or self._value_pages is None:
            return 0
        return self._key_pages.nbytes + self._value_pages.nbytes

    def validate_page_chunk(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("Paged K/V chunks must be 4D [1, H, S, D]")
        if key.shape != value.shape:
            raise ValueError("Paged K/V chunks must have the same shape")
        B, H, S, D = key.shape
        if B != 1:
            raise ValueError("Paged request cache only supports one request")
        if H <= 0 or D <= 0 or S <= 0:
            raise ValueError("Paged K/V chunks must have positive valid dimensions")
        if key.dtype != value.dtype or key.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "Paged K/V chunks must have the same float32 or bfloat16 dtype"
            )
        if key.device != value.device:
            raise ValueError("Paged K/V chunks must be on the same device")
        if (self._key_pages is None) != (self._value_pages is None):
            raise ValueError("Paged K/V storage is incomplete")
        if self._key_pages is not None and self._value_pages is not None:
            expected_shape = (H, self.page_size, D)
            if self._key_pages.shape[1:] != expected_shape:
                raise ValueError(
                    f"Paged K/V chunk shape {key.shape} does not match page storage "
                    f"shape {self._key_pages.shape}"
                )
            if (
                self._key_pages.dtype != key.dtype
                or self._value_pages.dtype != value.dtype
            ):
                raise ValueError(
                    "Paged K/V chunks must match the existing page storage dtype"
                )
            if self._key_pages.device != key.device:
                raise ValueError(
                    "Paged K/V chunks must match the existing page storage device"
                )

    def _snapshot_state(self) -> tuple:
        return (
            self._key_pages,
            self._value_pages,
            list(self.free_page_ids),
            set(self.used_page_ids),
            self.num_allocated_pages,
            self.reused_page_allocations,
            self.storage_growths,
            self.copied_pages_on_growth,
            self.copied_bytes_on_growth,
        )

    def _restore_state(self, state: tuple) -> None:
        (
            self._key_pages,
            self._value_pages,
            self.free_page_ids,
            self.used_page_ids,
            self.num_allocated_pages,
            self.reused_page_allocations,
            self.storage_growths,
            self.copied_pages_on_growth,
            self.copied_bytes_on_growth,
        ) = state

    def allocate_page(self) -> int:
        if self.free_page_ids:
            page_id = self.free_page_ids.pop()
            self.reused_page_allocations += 1
        else:
            page_id = self.num_pages
            self.num_allocated_pages += 1
        self.used_page_ids.add(page_id)
        return page_id

    def read_page(self, page_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._key_pages is None or self._value_pages is None:
            raise ValueError(f"Page {page_id} has no storage")
        if page_id >= self.num_pages:
            raise ValueError(f"Page {page_id} is out of range")
        return (
            self._key_pages[page_id : page_id + 1],
            self._value_pages[page_id : page_id + 1],
        )

    def _ensure_page_storage(self, key: torch.Tensor, value: torch.Tensor) -> None:
        B, H, _, D = key.shape
        assert B == 1

        if self._key_pages is not None and self._value_pages is not None:
            assert self._key_pages.shape[1:] == (H, self.page_size, D)
            assert self._value_pages.shape == self._key_pages.shape
            assert self._key_pages.dtype == key.dtype
            assert self._value_pages.dtype == value.dtype
            if self.capacity >= self.num_pages:
                return

        new_capacity = max(4, self.num_pages, self.capacity * 2)
        new_key_pages = torch.zeros(
            (new_capacity, H, self.page_size, D), dtype=key.dtype, device=key.device
        )
        new_value_pages = torch.zeros(
            (new_capacity, H, self.page_size, D), dtype=value.dtype, device=value.device
        )
        self.storage_growths += 1
        if self._key_pages is not None and self._value_pages is not None:
            old_pages = self.num_pages - 1
            self.copied_pages_on_growth += old_pages
            self.copied_bytes_on_growth += (
                self._key_pages[:old_pages].nbytes
                + self._value_pages[:old_pages].nbytes
            )
            new_key_pages[:old_pages, :, :, :] = self._key_pages[:old_pages]
            new_value_pages[:old_pages, :, :, :] = self._value_pages[:old_pages]
        self._key_pages = new_key_pages
        self._value_pages = new_value_pages

    def reset(self) -> None:
        if self.used_page_ids:
            raise ValueError("Cannot reset a page pool with live requests")
        self._key_pages = None
        self._value_pages = None
        self.free_page_ids.clear()
        self.num_allocated_pages = 0
        self.reused_page_allocations = 0
        self.storage_growths = 0
        self.copied_pages_on_growth = 0
        self.copied_bytes_on_growth = 0

    def write_page_slice(
        self,
        page_id: int,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.validate_page_chunk(key, value)
        if key.shape[2] > self.page_size:
            raise ValueError("Paged K/V writes cannot exceed one physical page")
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is free")
        if page_id < 0 or page_id >= self.num_pages:
            raise ValueError(f"Page {page_id} is out of range")
        end = start + key.shape[2]
        if start < 0 or end > self.page_size:
            raise ValueError("Paged K/V write is outside page storage")
        self._ensure_page_storage(key, value)
        assert self._key_pages is not None
        assert self._value_pages is not None
        H, capacity, D = self._key_pages.shape[1:]
        assert self._value_pages.shape == self._key_pages.shape
        assert capacity == self.page_size
        assert key.shape[:2] == (1, H)
        assert key.shape[3] == D
        assert 0 <= start <= capacity

        if key.is_cuda:
            self._key_pages = cuda_ops.paged_cache_update(
                self._key_pages,
                key.contiguous(),
                page_id,
                start,
            )
            self._value_pages = cuda_ops.paged_cache_update(
                self._value_pages,
                value.contiguous(),
                page_id,
                start,
            )
        else:
            self._key_pages[page_id, :, start:end, :].copy_(key[0])
            self._value_pages[page_id, :, start:end, :].copy_(value[0])

    def free_page(self, page_id: int) -> None:
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is already free")
        self.used_page_ids.remove(page_id)
        self.free_page_ids.append(page_id)


class TinyKvPagedCache(TinyKvCache):
    """Request-and-layer-local K/V cache backed by a layer page pool."""

    def __init__(self, pool: TinyKvPagedPool):
        self.pool = pool
        self.page_size = self.pool.page_size
        self.page_ids: list[int] = []
        self.page_lens: list[int] = []
        self.offset = 0
        self._cached_block_table: torch.Tensor | None = None
        self._cached_block_table_key: tuple[tuple[int, ...], int, str] | None = None

    @property
    def num_pages(self) -> int:
        return len(self.page_ids)

    @property
    def key_values(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.offset == 0:
            return None
        return self.gather_dense()

    def _append_chunk(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.pool.validate_page_chunk(key, value)
        S = key.shape[2]
        cache_state = self._snapshot_state()
        pool_state = self.pool._snapshot_state()
        start = 0

        try:
            if self.page_ids and self.page_lens[-1] < self.page_size:
                page_id = self.page_ids[-1]
                page_start = self.page_lens[-1]
                take = min(self.page_size - page_start, S)
                self.pool.write_page_slice(
                    page_id,
                    page_start,
                    key[:, :, :take, :],
                    value[:, :, :take, :],
                )
                self.page_lens[-1] += take
                start += take

            while start < S:
                end = min(start + self.page_size, S)
                page_id = self.pool.allocate_page()
                self.pool.write_page_slice(
                    page_id,
                    0,
                    key[:, :, start:end, :],
                    value[:, :, start:end, :],
                )
                self.page_ids.append(page_id)
                self.page_lens.append(end - start)
                start = end

            self.offset += S
        except Exception:
            self.pool._restore_state(pool_state)
            self._restore_state(cache_state)
            raise

    def validate_append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.pool.validate_page_chunk(key, value)

    def _snapshot_state(self) -> tuple:
        return (
            list(self.page_ids),
            list(self.page_lens),
            self.offset,
            self._cached_block_table,
            self._cached_block_table_key,
        )

    def _restore_state(self, state: tuple) -> None:
        (
            self.page_ids,
            self.page_lens,
            self.offset,
            self._cached_block_table,
            self._cached_block_table_key,
        ) = state

    def gather_dense(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.offset > 0
        key_chunks = []
        value_chunks = []
        for page_id, page_len in zip(self.page_ids, self.page_lens):
            key_page, value_page = self.pool.read_page(page_id)
            assert key_page.shape[2] == self.page_size
            assert value_page.shape[2] == self.page_size
            key_chunks.append(key_page[:, :, :page_len, :])
            value_chunks.append(value_page[:, :, :page_len, :])
        if len(key_chunks) == 1:
            return key_chunks[0], value_chunks[0]
        return torch.cat(key_chunks, dim=2), torch.cat(value_chunks, dim=2)

    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        self._append_chunk(key, value)
        dense_key, dense_value = self.gather_dense()
        return dense_key, dense_value, self.offset, mask

    def _device(self) -> torch.device:
        if self.pool._key_pages is not None:
            return self.pool._key_pages.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def block_table(self, max_pages: int | None = None) -> torch.Tensor:
        if max_pages is None:
            max_pages = self.num_pages
        assert max_pages >= self.num_pages
        device = self._device()
        cache_key = (tuple(self.page_ids), max_pages, str(device))
        if (
            self._cached_block_table is not None
            and self._cached_block_table_key == cache_key
        ):
            return self._cached_block_table
        page_ids = self.page_ids + [-1] * (max_pages - self.num_pages)
        self._cached_block_table = torch.tensor(
            [page_ids], dtype=torch.int32, device=device
        )
        self._cached_block_table_key = cache_key
        return self._cached_block_table

    def context_lens(self) -> torch.Tensor:
        return torch.tensor([self.offset], dtype=torch.int32, device=self._device())

    def paged_metadata(
        self,
        max_pages: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> PagedKvMetadata:
        assert self.pool.key_pages is not None
        assert self.pool.value_pages is not None
        return PagedKvMetadata(
            key_pages=self.pool.key_pages,
            value_pages=self.pool.value_pages,
            block_table=self.block_table(max_pages=max_pages),
            context_lens=self.context_lens(),
            page_size=self.page_size,
            mask=mask,
        )

    def update_and_fetch_paged(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> PagedKvMetadata:
        self._append_chunk(key, value)
        return self.paged_metadata(mask=mask)

    def materialize(self):
        key_pages = self.pool.key_pages
        if key_pages is not None and key_pages.is_cuda:
            torch.cuda.synchronize(key_pages.device)

    def rewind(self, n: int):
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= self.offset:
            raise ValueError("rewind length must be between zero and the cache length")
        new_offset = self.offset - n
        if new_offset == self.offset:
            return
        if new_offset == 0:
            self.release()
            return

        target_num_pages = (new_offset + self.page_size - 1) // self.page_size
        while len(self.page_ids) > target_num_pages:
            page_id = self.page_ids.pop()
            self.page_lens.pop()
            self.pool.free_page(page_id)

        last_page_len = new_offset - self.page_size * (target_num_pages - 1)
        self.page_lens[-1] = last_page_len
        self.offset = new_offset

    def release(self):
        for page_id in self.page_ids:
            self.pool.free_page(page_id)
        self.page_ids.clear()
        self.page_lens.clear()
        self.offset = 0
