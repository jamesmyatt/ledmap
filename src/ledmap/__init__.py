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
