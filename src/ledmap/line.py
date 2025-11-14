"""1D Line of Pixels."""

from typing import Literal

import numpy as np

FirstPixel1D = Literal["start", "end"]


def check_shape(
    length: int = -1,
    leds: np.ndarray | None = None,
) -> int:
    """Check/infer shape of line."""
    if length <= 0:
        if leds is None:
            msg = "number or leds must be provided."
            raise ValueError(msg)
        length = leds.size

    return length


def make_line(
    length: int = -1,
    *,
    leds: np.ndarray | None = None,
    first: FirstPixel1D = "start",
    fill_value: int = -1,
) -> np.ndarray:
    """Prepare line of pixels of specific length."""
    length = check_shape(length, leds)

    if leds is None:
        leds = np.arange(length, dtype=int)
    else:
        leds = np.copy(leds).flatten()
        if leds.size < length:
            # Pad after
            leds[leds.size : length] = fill_value
        elif leds.size > length:
            # Truncate
            leds = leds[:length]

    # Reverse list
    if first == "end":
        leds = leds[::-1]  # np.flip(..., axis=0)

    return leds
