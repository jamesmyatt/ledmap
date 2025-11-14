"""2D matrix of pixels."""

from typing import Literal

import numpy as np

from .line import make_line

FirstPixel2D = Literal["top-left", "top-right", "bottom-left", "bottom-right"]


def check_shape(
    height: int = -1,
    width: int = -1,
    pixels: np.ndarray | None = None,
) -> tuple[int, int]:
    """Check/infer shape of matrix."""
    msg = "At least 2 of height, width and leds must be provided."
    if pixels is None:
        if width <= 0 or height <= 0:
            raise ValueError(msg)
    else:
        if width <= 0 and height <= 0:
            raise ValueError(msg)

        # Calculate missing dimension by rounding-up
        if width <= 0:
            width = -(-pixels.size // height)
        elif height <= 0:
            height = -(-pixels.size // width)

    return height, width


def make_matrix(
    height: int = -1,
    width: int = -1,
    *,
    pixels: np.ndarray | None = None,
    serpentine: bool = False,
    vertical: bool = False,
    first: FirstPixel2D = "top-left",
    fill_value: int = -1,
) -> np.ndarray:
    """Generate map for simple matrix."""
    height, width = check_shape(height, width, pixels)
    pixels = make_line(
        width * height, pixels=pixels, first="start", fill_value=fill_value
    )

    order = "F" if vertical else "C"
    matrix = pixels.reshape((height, width), order=order)

    if serpentine:
        matrix = __serpentine(matrix, vertical=vertical)

    if first.lower() != "top-left":
        matrix = reorient(matrix, first)

    return matrix


def reorient(matrix: np.ndarray, first: FirstPixel2D = "top-left") -> np.ndarray:
    """Reorient matrix by new position of first (top-left) pixel.

    Maintains direction of the sequence (i.e. horizontal or vertical).
    """
    assert matrix.ndim == 2
    origin = first.lower().split("-", 1)
    if origin[0] == "bottom":
        matrix = np.flipud(matrix)
    if origin[1] == "right":
        matrix = np.fliplr(matrix)
    return matrix


def serpentine(matrix: np.ndarray, *, vertical: bool = False) -> np.ndarray:
    """Make matrix serpentine."""
    assert matrix.ndim == 2

    matrix = np.copy(matrix)
    if vertical:
        matrix[:, 1::2] = np.flip(matrix[:, 1::2], axis=0)
    else:
        matrix[1::2, :] = np.flip(matrix[1::2, :], axis=1)

    return matrix


# Make private version to avoid bypass with argument names
__serpentine = serpentine
