"""2D Matrix of Pixels."""

from typing import Literal

import numpy as np

from .line import make_line

FirstPixel2D = Literal["top-left", "top-right", "bottom-left", "bottom-right"]


def check_shape(
    height: int = -1,
    width: int = -1,
    leds: np.ndarray | None = None,
) -> tuple[int, int]:
    """Check/infer shape of matrix."""
    msg = "At least 2 of height, width and leds must be provided."
    if leds is None:
        if width <= 0 or height <= 0:
            raise ValueError(msg)
    else:
        if width <= 0 and height <= 0:
            raise ValueError(msg)

        # Calculate missing dimension by rounding-up
        if width <= 0:
            width = -(-leds.size // height)
        elif height <= 0:
            height = -(-leds.size // width)

    return height, width


def make_matrix(
    height: int = -1,
    width: int = -1,
    *,
    leds: np.ndarray | None = None,
    serpentine: bool = False,
    vertical: bool = False,
    first: FirstPixel2D = "top-left",
    fill_value: int = -1,
) -> np.ndarray:
    """Generate map for simple matrix."""
    height, width = check_shape(height, width, leds)
    leds = make_line(width * height, leds=leds, first="start", fill_value=fill_value)

    order = "F" if vertical else "C"
    array = leds.reshape((height, width), order=order)

    if serpentine:
        array = __serpentine(array, vertical=vertical)

    if first.lower() != "top-left":
        array = orient_matrix(array, first)

    return array


def orient_matrix(array: np.ndarray, first: FirstPixel2D = "top-left") -> np.ndarray:
    """Orient matrix by position of first.

    Maintains direction of the sequence (e.g. horizontal or vertical).
    """
    assert array.ndim == 2
    origin = first.lower().split("-", 1)
    if origin[0] == "bottom":
        array = np.flipud(array)
    if origin[1] == "right":
        array = np.fliplr(array)
    return array


def serpentine(array: np.ndarray, *, vertical: bool = False) -> np.ndarray:
    """Make matrix serpentine."""
    assert array.ndim == 2

    array = np.copy(array)
    if vertical:
        array[:, 1::2] = np.flip(array[:, 1::2], axis=0)
    else:
        array[1::2, :] = np.flip(array[1::2, :], axis=1)

    return array


# Make private version to avoid bypass with argument names
__serpentine = serpentine
