"""tqdm progress helper."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


def progress_iter(iterable: Iterable[T], *, total: Optional[int] = None, desc: str = "progress", every: int = 10) -> Iterator[T]:
    """Yield items with a tqdm progress bar."""
    yield from tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
