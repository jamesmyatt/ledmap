"""Pixel mappings."""

import itertools
from collections.abc import Iterable

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
