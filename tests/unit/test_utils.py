import e3nn_jax as e3j
import pytest

from tensorial import utils


def test_hidden_irreps():
    assert utils.hidden_irreps(8, 2) == "8x0o + 8x0e + 8x1o + 8x1e + 8x2o + 8x2e"
    assert utils.hidden_irreps(16, 1) == "16x0o + 16x0e + 16x1o + 16x1e"


def test_hidden_irreps_lmax_zero():
    assert utils.hidden_irreps(4, 0) == "4x0o + 4x0e"


def test_hidden_irreps_is_valid_irreps():
    irreps = e3j.Irreps(utils.hidden_irreps(8, 3))
    assert irreps.dim == 8 * sum(2 * ell + 1 for ell in range(4)) * 2
    assert irreps.lmax == 3


def test_hidden_irreps_rejects_negative():
    with pytest.raises(ValueError):
        utils.hidden_irreps(-1, 2)

    with pytest.raises(ValueError):
        utils.hidden_irreps(8, -1)
