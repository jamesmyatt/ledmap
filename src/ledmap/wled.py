import io
import json

import numpy as np

from . import make_array


def to_dict(mapping: np.ndarray, *, name: str = "") -> dict:
    """Make WLED ledmap dictionary."""
    out = {}
    match mapping.ndim:
        case 1:
            out["map"] = mapping.flatten().tolist()
        case 2:
            out["map"] = mapping.flatten().tolist()
            out["width"] = mapping.shape[1]
            out["height"] = mapping.shape[0]

    # Extra attributes for MM
    if name:
        out["n"] = name
    return out


def dump(mapping: dict | np.ndarray, f: io.TextIOBase) -> None:
    """Dump WLED ledmap file."""
    if not isinstance(mapping, dict):
        mapping = to_dict(mapping)
    json.dump(mapping, f, indent=None, separators=(",", ":"))


def from_dict(mapping: dict) -> np.ndarray:
    """Make LED mapping array."""
    return make_array(
        mapping["map"],
        width=mapping.get("width", -1),
        height=mapping.get("height", -1),
    )
