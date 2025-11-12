from collections.abc import Callable

import pytest

from ledmap import matrix

WRAPPERS = [
    None,
    matrix.Wrapper,
    matrix.FlipLR,
    matrix.FlipUD,
    matrix.Serpentine,
    matrix.Transpose,
    matrix.Rot90,
    matrix.Rot180,
    matrix.Rot270,
    matrix.Limit,
]


def chain_wrappers(
    matrix: matrix.Base,
    *wrappers: Callable[[matrix.Base], matrix.Base] | None,
) -> matrix.Base:
    for wrapper in wrappers:
        if wrapper is not None:
            matrix = wrapper(matrix)
    return matrix


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_length(wrapper):
    m = chain_wrappers(matrix.Matrix(3, 4), wrapper)

    assert sum(1 for _ in m) == 12
    assert sum(1 for _ in m.iter()) == 12

    assert isinstance(m.map, tuple)
    assert len(m.map) == 12


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_shape(wrapper):
    m = chain_wrappers(matrix.Matrix(3, 4), wrapper)
    shape = (m.height, m.width)  # Use NumPy convention

    transpose = wrapper is not None and issubclass(wrapper, matrix.Transpose)
    shape_exp = (3, 4) if transpose else (4, 3)

    assert shape == shape_exp
    assert len(tuple(m.row(1))) == shape_exp[1]
    assert len(tuple(m.column(1))) == shape_exp[0]


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_custom_round_trip(wrapper):
    m = chain_wrappers(matrix.Matrix(3, 4), wrapper)
    c = matrix.Custom(**m.ledmap())
    assert m == c
