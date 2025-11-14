"""1D line of pixels."""

from typing import Literal

import numpy as np

FirstPixel1D = Literal["start", "end"]


def check_shape(
    length: int = -1,
    pixels: np.ndarray | None = None,
) -> int:
    """Check/infer shape of line."""
    if length <= 0:
        if pixels is None:
            msg = "number or leds must be provided."
            raise ValueError(msg)
        length = pixels.size

    return length


def make_line(
    length: int = -1,
    *,
    pixels: np.ndarray | None = None,
    first: FirstPixel1D = "start",
    fill_value: int = -1,
) -> np.ndarray:
    """Prepare line of pixels of specific length."""
    length = check_shape(length, pixels)

    if pixels is None:
        pixels = np.arange(length, dtype=int)
    else:
        pixels = np.copy(pixels).flatten()
        if pixels.size < length:
            # Pad after
            pixels[pixels.size : length] = fill_value
        elif pixels.size > length:
            # Truncate
            pixels = pixels[:length]

    # Reverse list
    if first == "end":
        pixels = pixels[::-1]  # np.flip(..., axis=0)

    return pixels
