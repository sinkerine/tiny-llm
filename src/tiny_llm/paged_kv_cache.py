from dataclasses import dataclass
from typing import Optional

import torch

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
    """Layer-local physical storage for paged KV."""

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
        return None if self._key_pages is None else self._key_pages[: self.num_pages]

    @property
    def value_pages(self) -> torch.Tensor | None:
        return None if self._value_pages is None else self._value_pages[: self.num_pages]

    @property
    def capacity(self) -> int:
        return 0 if self._key_pages is None else self._key_pages.shape[0]

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

    def allocate_page(self) -> int:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedPool")

    def read_page(self, page_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedPool")

    def write_page_slice(
        self, page_id: int, start: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedPool")

    def free_page(self, page_id: int) -> None:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedPool")

    def reset(self) -> None:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedPool")


class TinyKvPagedCache(TinyKvCache):
    """Request-and-layer-local K/V cache backed by a layer page pool."""

    def __init__(self, pool: TinyKvPagedPool):
        self.pool = pool
        self.page_size = self.pool.page_size
        self.page_ids: list[int] = []
        self.page_lens: list[int] = []
        self.offset = 0

    @property
    def num_pages(self) -> int:
        return len(self.page_ids)

    @property
    def key_values(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.offset == 0:
            return None
        return self.gather_dense()

    def gather_dense(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedCache")

    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedCache")

    def block_table(self, max_pages: int | None = None) -> torch.Tensor:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedCache")

    def context_lens(self) -> torch.Tensor:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedCache")

    def paged_metadata(
        self, max_pages: int | None = None, mask: torch.Tensor | str | None = None
    ) -> PagedKvMetadata:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedCache")

    def update_and_fetch_paged(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> PagedKvMetadata:
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedCache")

    def rewind(self, n: int):
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedCache.rewind")

    def release(self):
        raise NotImplementedError("Week 3 Day 3: implement TinyKvPagedCache.release")
