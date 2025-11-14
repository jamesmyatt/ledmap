"""Legacy pixel mapping.

From v0.1. To be kept until replaced.
"""

import io
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from . import wled
from .pixels import as_string, make_array


class Base(ABC):
    """2D pixel mapper base."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Width of the matrix."""
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        """Height of the matrix."""
        ...

    def get(self, x: int, y: int) -> int:
        """Index of the pixel with checks."""
        assert x in range(self.width)
        assert y in range(self.height)
        return self.mapper(x, y)

    @abstractmethod
    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        ...

    @property
    def map(self) -> tuple[int, ...]:
        """Indices of all pixels."""
        return tuple(self)

    def __iter__(self) -> Iterator[int]:
        """Iterate over indices of each pixel."""
        yield from self.iter()

    def iter(self) -> Iterator[int]:
        """Iterate over indices of each pixel."""
        for y in range(self.height):
            for x in range(self.width):
                yield self.mapper(x, y)

    def row(self, y: int) -> Iterator[int]:
        """Iterate over indices of each pixel in row."""
        for x in range(self.width):
            yield self.mapper(x, y)

    def column(self, x: int) -> Iterator[int]:
        """Iterate over indices of each pixel in column."""
        for y in range(self.height):
            yield self.mapper(x, y)

    def to_array(self) -> np.ndarray:
        """Convert to 2D NumPy array."""
        return make_array(self.map, width=self.width)

    def ledmap(self) -> dict:
        """WLED ledmap information."""
        return wled.from_array(self.to_array())

    def dump(self, f: io.TextIOBase) -> None:
        """Write WLED ledmap as compact JSON."""
        return wled.dump(self.to_array(), f)

    def as_string(self, **kwargs: Any) -> str:
        """Convert to string representation."""
        return as_string(self.to_array(), **kwargs)

    def __str__(self) -> str:
        """Render as string."""
        return self.as_string()

    def print(self, **kwargs: Any) -> None:
        """Print information about mapping."""
        print(repr(self))
        print(self.as_string(**{"prefix": "  ", "sep": "  ", **kwargs}))

    def __eq__(self, rhs) -> bool:
        """Check for equality of matrices."""
        return (
            isinstance(self, Base)
            and isinstance(rhs, Base)
            and self.width == rhs.width
            and self.height == rhs.height
            and self.map == rhs.map
        )

    def repr_args(self) -> tuple[list[str], dict[str, str]]:
        """Get arguments to constructor."""
        return [], {}

    def __repr__(self) -> str:
        """Representation of object."""
        args, kwargs = self.repr_args()
        args.extend(f"{k}={v}" for k, v in kwargs.items())
        return f"{self.__class__.__name__}({','.join(args)})"


class Matrix(Base):
    """Standard matrix."""

    def __init__(self, width: int = 1, height: int = 1):
        """Create standard matrix."""
        assert width > 0
        assert height > 0
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        """Width of the matrix."""
        return self._width

    @property
    def height(self) -> int:
        """Height of the matrix."""
        return self._height

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        # Default mapper
        return x + y * self.width

    def __repr__(self) -> str:
        """Representation of object."""
        return (
            f"{self.__class__.__name__}(width={self._width!r},height={self._height!r})"
        )


class Wrapper(Base):
    """Identity wrapper."""

    def __init__(self, mapper: Base):
        """Create wrapper for matrix."""
        self._wraps = mapper

    @property
    def width(self) -> int:
        """Width of the matrix."""
        return self._wraps.width

    @property
    def height(self) -> int:
        """Height of the matrix."""
        return self._wraps.height

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        return self._wraps.mapper(x, y)

    def repr_args(self) -> tuple[list[str], dict[str, str]]:
        """Get arguments to constructor."""
        args, kwargs = super().repr_args()
        args.append(repr(self._wraps))
        return args, kwargs


class FlipLR(Wrapper):
    """Flip left-right."""

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        return self._wraps.mapper(self.width - x - 1, y)


class FlipUD(Wrapper):
    """Flip upside-down."""

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        return self._wraps.mapper(x, self.height - y - 1)


class Rot180(Wrapper):
    """Rotate half-turn."""

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        return self._wraps.mapper(self.width - x - 1, self.height - y - 1)


class Serpentine(Wrapper):
    """Flip alternate rows."""

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        x_ = self.width - x - 1 if y % 2 else x
        return self._wraps.mapper(x_, y)


class Transpose(Wrapper):
    """Swap rows and columns."""

    @property
    def width(self) -> int:
        """Width of the matrix."""
        return self._wraps.height

    @property
    def height(self) -> int:
        """Height of the matrix."""
        return self._wraps.width

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        return self._wraps.mapper(y, x)


class Rot90(Transpose):
    """Rotate quarter-turn anti-clockwise.

    Matches numpy.rot90.
    """

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        return self._wraps.mapper(self.height - y - 1, x)


class Rot270(Transpose):
    """Rotate quarter-turn clockwise.

    Matches numpy.rot90 with k=3.
    """

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        return self._wraps.mapper(y, self.width - x - 1)


class Limit(Wrapper):
    """Limit pixel indices."""

    def __init__(
        self,
        mapper: Base,
        first: int = 0,
        last: int | None = None,
    ):
        """Create wrapper for matrix."""
        super().__init__(mapper=mapper)

        assert first >= 0
        assert last is None or last > 0
        self._first = first
        self._last = last

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        index = self._wraps.mapper(x, y)
        if index < self._first:
            index = -1
        if self._last is not None and index > self._last:
            index = -1
        return index

    def repr_args(self) -> tuple[list[str], dict[str, str]]:
        """Get arguments to constructor."""
        args, kwargs = super().repr_args()
        if self._first > 0:
            kwargs["first"] = repr(self._first)
        if self._last is not None:
            kwargs["last"] = repr(self._last)
        return args, kwargs


class Custom(Base):
    """Custom matrix."""

    def __init__(
        self,
        map: Iterable[int],  # noqa: A002
        width: int = 1,
        height: int | None = None,
    ):
        """Create custom matrix."""
        assert width > 0
        assert height is None or height > 0
        self._map: tuple[int, ...] = tuple(map)
        self._width = width
        self._height = -(-len(self._map) // self._width) if height is None else height

    @property
    def width(self) -> int:
        """Width of the matrix."""
        return self._width

    @property
    def height(self) -> int:
        """Height of the matrix."""
        return self._height

    def mapper(self, x: int, y: int) -> int:
        """Map pixel location to index."""
        # Custom mapper
        index = x + y * self.width
        try:
            return self._map[index]
        except IndexError:
            return -1

    def repr_args(self) -> tuple[list[str], dict[str, str]]:
        """Get arguments to repr."""
        args, kwargs = super().repr_args()
        kwargs["map"] = f"[{','.join(str(i) for i in self._map)}]"
        kwargs["width"] = repr(self._width)
        kwargs["height"] = repr(self._height)
        return args, kwargs
