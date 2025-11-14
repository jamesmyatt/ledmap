"""Pixel mappings."""

import itertools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Concatenate

import numpy as np


def make_array(
    map: Iterable[int],  # noqa: A002
    width: int = -1,
    height: int = -1,
) -> np.ndarray:
    """Make LED mapping array."""
    array = np.array(map, dtype=int)

    if width > 0 or height > 0:
        shape = (
            height if height > 0 else -1,
            width if width > 0 else -1,
        )
        array = array.reshape(shape)

    return array


def as_string(
    array: np.ndarray,
    *,
    sep: str = ", ",
    prefix: str = "",
    postfix: str = "",
    missing: str = "-1",
    width: int = 0,
) -> str:
    """Convert to string representation."""
    strings = [str(i) if i >= 0 else missing for i in array.flat]
    n = max(width, max(len(s) for s in strings))

    match array.ndim:
        case 1:
            return prefix + sep.join(f"{s:>{n}}" for s in strings) + postfix
        case 2:
            cols = array.shape[1]
            return "\n".join(
                prefix + sep.join(f"{s:>{n}}" for s in row) + postfix
                for row in itertools.batched(strings, cols, strict=True)
            )

    msg = "Shape not supported"
    raise ValueError(msg)


@dataclass
class Mapping:
    """Pixel map from mapping array."""

    array: np.ndarray

    @classmethod
    def from_shape(cls, shape: tuple[int, ...], *, vertical: bool = False):
        """Create default mapping from shape."""
        if not all(s > 0 for s in shape):
            msg = "All dimensions must be strictly positive."
            raise ValueError(msg)

        length = np.prod(shape, dtype=int)
        order = "F" if vertical else "C"
        array = np.arange(length, int).reshape(shape, order=order)

        return cls(array)

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of array."""
        return self.array.shape

    def apply(
        self,
        func: Callable[Concatenate[np.ndarray, ...], np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> "Mapping":
        """Apply function to mapping array."""
        array = func(self.array, *args, **kwargs)
        return Mapping(array)


@dataclass
class Mapper:
    """Pixel map from mapping function."""

    shape: tuple[int, ...]

    def index(self) -> Iterator[tuple[int, ...]]:
        """Iterate over location indices."""
        ranges = (range(n) for n in self.shape)
        return itertools.product(*ranges)

    def iter(self) -> Iterator[int]:
        """Iterate over pixel indices."""
        for loc in self.index():
            yield self.get(*loc)

    def __iter__(self) -> Iterator[int]:
        """Iterate over pixel indices."""
        yield from self.iter()

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.ndarray(list(self), int).reshape(self.shape)

    def to_mapping(self) -> Mapping:
        """Convert to mapping object."""
        return Mapping(self.to_array())

    def get(self, *loc: int) -> int:
        """Get pixel index at location.

        Override this method for different pixel mappings.

        Default is row-major index.
        """
        out: int = 0
        k: int = 1
        for i, n in zip(loc, self.shape, strict=True):
            assert 0 <= i < n
            out += i * k
            k *= n
        return out
