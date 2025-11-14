"""Arbitrary utilities, including for notebooks."""

import io
from collections.abc import Callable


def get_string(func: Callable[[io.TextIOBase], None]) -> str:
    """Capture output stream as a string."""
    with io.StringIO() as f:
        func(f)
        return f.getvalue()
