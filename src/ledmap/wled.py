"""WLED interations."""

import io
import json
from typing import Any

import numpy as np

from .pixels import make_array as _make_array


def from_array(array: np.ndarray, *, name: str = "") -> dict[str, Any]:
    """Make WLED ledmap dictionary."""
    # Create 1D array with missing values fixed
    map_ = array.flatten()
    map_ = np.where(map_ >= 0, map_, -1).astype(int)

    # Construct data structure
    out: dict[str, Any] = {
        "map": map_.tolist(),
    }
    match array.ndim:
        case 1:
            pass
        case 2:
            out["width"] = array.shape[1]
            out["height"] = array.shape[0]
        case _:
            msg = "Mapping must have 1 or 2 dimensions."
            raise ValueError(msg)

    # Extra attributes for MM
    if name:
        out["n"] = name
    return out


def dump(mapping: dict | np.ndarray, f: io.TextIOBase) -> None:
    """Dump WLED ledmap file."""
    if not isinstance(mapping, dict):
        mapping = from_array(mapping)
    json.dump(mapping, f, indent=None, separators=(",", ":"))


def to_array(ledmap: dict) -> np.ndarray:
    """Make pixel mapping array."""
    shape = (
        ledmap.get("height", -1),
        ledmap.get("width", -1),
    )
    if all(s <= 0 for s in shape):
        shape = ()

    return _make_array(ledmap["map"], shape=shape)
