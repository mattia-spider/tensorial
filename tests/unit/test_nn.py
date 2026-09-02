from flax import linen
import jax
import jax.numpy as jnp
import pytest

from tensorial import nn


class Add(linen.Module):
    amount: float

    @linen.compact
    def __call__(self, x):
        return x + self.amount


def _apply(model, value=0.0):
    x = jnp.array([value])
    params = model.init(jax.random.PRNGKey(0), x)
    return float(model.apply(params, x)[0])


def test_sequential_splices_nested_lists():
    """A layers entry that is itself a list of layers is spliced into the chain, so a factory
    function can be dropped straight into a config alongside hand-written layers."""
    model = nn.Sequential([Add(1.0), [Add(2.0), Add(3.0)], Add(4.0)])

    assert len(nn._layers(model.layers)) == 4  # pylint: disable=protected-access
    assert _apply(model) == 10.0


def test_sequential_splices_deeply_nested_lists():
    model = nn.Sequential([Add(1.0), [[Add(2.0)], (Add(3.0),)]])

    assert len(nn._layers(model.layers)) == 3  # pylint: disable=protected-access
    assert _apply(model) == 6.0


def test_sequential_keeps_nested_sequential_nested():
    """Nested `Sequential`s are modules in their own right, flattening them would change the
    parameter scoping of existing models."""
    model = nn.Sequential([Add(1.0), nn.Sequential([Add(2.0), Add(3.0)])])

    assert len(nn._layers(model.layers)) == 2  # pylint: disable=protected-access
    assert _apply(model) == 6.0


def test_sequential_rejects_empty():
    with pytest.raises(ValueError):
        nn.Sequential([])


def test_sequential_rejects_non_sequence():
    with pytest.raises(ValueError):
        nn.Sequential(Add(1.0))
