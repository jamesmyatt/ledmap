"""Pixel mappings."""

import io
import itertools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Concatenate

import numpy as np


def make_array(
    pixels: int | Iterable[int] | np.ndarray,
    *,
    shape: tuple[int, ...] = (),
    vertical: bool = False,
) -> np.ndarray:
    """Make LED mapping array."""
    if isinstance(pixels, int):
        array = np.arange(pixels, dtype=int)
    else:
        array = np.asanyarray(pixels, dtype=int)

    if shape:
        order = "F" if vertical else "C"
        array = array.reshape(shape, order=order)

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
    def from_list(
        cls,
        pixels: Iterable[int],
        *,
        shape: tuple[int, ...] = (),
        vertical: bool = False,
    ) -> "Mapping":
        """Make LED mapping array."""
        array = make_array(pixels, shape=shape, vertical=vertical)
        return cls(array)

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        *,
        vertical: bool = False,
    ):
        """Create default mapping from shape."""
        if not all(s > 0 for s in shape):
            msg = "All dimensions must be strictly positive."
            raise ValueError(msg)

        length = np.prod(shape, dtype=int).item()
        array = make_array(length, shape=shape, vertical=vertical)

        return cls(array)

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of array."""
        return self.array.shape

    def __iter__(self) -> Iterable[int]:
        """Iterate over pixel indices."""
        yield from self.array.flat

    def apply(
        self,
        func: Callable[Concatenate[np.ndarray, ...], np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> "Mapping":
        """Apply function to mapping array."""
        array = func(self.array, *args, **kwargs)
        return Mapping(array)

    def reshape(self, shape: tuple[int, ...]) -> "Mapping":
        """Reshape pixel array."""
        # TODO: implement padding/truncating
        return self.apply(np.reshape, shape=shape)

    def as_string(self, **kwargs: Any) -> str:
        """Convert to string representation."""
        return as_string(self.array, **kwargs)

    def __str__(self) -> str:
        """Render as string."""
        return self.as_string()

    def print(self, **kwargs: Any) -> None:
        """Print information about mapping."""
        print(self.as_string(**{"prefix": "  ", "sep": "  ", **kwargs}))

    def equals(self, rhs: "Mapping | np.ndarray") -> bool:
        """Check for equality of mappings."""
        if isinstance(rhs, Mapping):
            rhs = rhs.array
        return np.array_equal(self.array, rhs, equal_nan=True)

    def to_wled(self) -> dict[str, Any]:
        """WLED ledmap information."""
        from .wled import from_array as _from_array

        return _from_array(self.array)

    def dump_wled(self, f: io.TextIOBase) -> None:
        """Write WLED ledmap file."""
        from .wled import dump as _dump

        return _dump(self.array, f)

    def summary(self) -> str:
        """Summary text."""
        return (
            f"{'x'.join(str(n) for n in self.shape)} pixel"
            f"{'' if self.array.size == 1 else 's'}"
        )

    def _repr_pretty_(self, p, cycle: bool = False) -> None:  # noqa: ARG002, FBT001, FBT002
        """Pretty printer for IPython."""
        p.text(self.summary())
        p.text(self.as_string())

    def _repr_html_(self) -> str:
        out = f"<p>{self.summary()}</p>"
        match self.ndim:
            case 1:
                out += (
                    "<table><tr>"
                    + "".join(
                        "<tr>"
                        + "".join(
                            f'<td style="border: 1px solid black;">{i}</td>'
                            for i in row
                        )
                        + "</tr>"
                        for row in itertools.batched(self.array.flat, 20, strict=False)
                    )
                    + "</tr><table>"
                )
            case 2:
                out += (
                    "<table>"
                    + "".join(
                        "<tr>"
                        + "".join(
                            f'<td style="border: 1px solid black;">{i}</td>'
                            for i in row
                        )
                        + "</tr>"
                        for row in self.array
                    )
                    + "<table>"
                )
            case _:
                out += f"<pre>{self!r}</pre>"
        return out


@dataclass
class Mapper:
    """Pixel map from mapping function."""

    shape: tuple[int, ...]

    @property
    def length(self) -> int:
        """Size of array."""
        return np.prod(self.shape, dtype=int).item()

    def summary(self) -> str:
        """Summary text."""
        return (
            f"{'x'.join(str(n) for n in self.shape)} pixel"
            f"{'' if self.length == 1 else 's'}"
        )

    def check(self) -> None:
        """Check mapper parameters."""
        msg = None
        if not self.shape:
            msg = "Shape must not be empty."
        elif any(s <= 0 for s in self.shape):
            msg = "All dimensions must have size > 0."
        if msg:
            raise ValueError(msg)

    def index(self) -> Iterator[tuple[int, ...]]:
        """Iterate over location indices."""
        ranges = (range(n) for n in self.shape)
        return itertools.product(*ranges)

    def iter(self) -> Iterator[int]:
        """Iterate over pixel indices."""
        self.check()
        for loc in self.index():
            yield self.get(*loc)

    def __iter__(self) -> Iterator[int]:
        """Iterate over pixel indices."""
        yield from self.iter()

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array(list(self), dtype=int).reshape(self.shape)

    def to_mapping(self) -> Mapping:
        """Convert to mapping object."""
        return Mapping(self.to_array())

    def get(self, *loc: int) -> int:
        """Get pixel index at location.

        Override this method for different pixel mappings.

        Default is row-major index.
        """
        return _ravel_index_C(loc, self.shape)


def _ravel_index_C(index: tuple[int, ...], shape: tuple[int, ...]) -> int:  # noqa: N802
    # TODO: This could be more efficient, or use numpy
    out: int = 0
    k: int = 1
    for i, n in zip(index, shape, strict=True):
        if not 0 <= i < n:
            msg = "Index out of range"
            raise ValueError(msg)
        out = (k * out) + i
        k = n
    return out
