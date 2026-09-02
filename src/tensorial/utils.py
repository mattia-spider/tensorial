import importlib
import types

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import numpy as np

from tensorial.typing import IntoIrreps


def infer_backend(pytree) -> types.ModuleType:
    """Try to infer a backend from the passed pytree"""
    any_numpy = any(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(pytree))
    any_jax = any(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(pytree))
    if any_numpy and any_jax:
        raise ValueError("Cannot mix numpy and jax arrays")

    if any_numpy:
        return np

    if any_jax:
        return jnp

    return jnp


def hidden_irreps(mul: int, lmax: int) -> str:
    r"""Build the irreps string for a stack of hidden features.

    Contains ``mul`` copies of every irrep up to (and including) ``lmax``, in both parities, odd
    before even, e.g. ``hidden_irreps(8, 2)`` gives
    ``"8x0o + 8x0e + 8x1o + 8x1e + 8x2o + 8x2e"``.

    Args:
        mul: the multiplicity (number of channels) of each irrep
        lmax: the maximum rotation order to include

    Returns:
        the irreps as a string, suitable for passing to :class:`e3nn_jax.Irreps`
    """
    if mul < 0:
        raise ValueError(f"'mul' must be non-negative, got {mul}")
    if lmax < 0:
        raise ValueError(f"'lmax' must be non-negative, got {lmax}")

    return " + ".join(f"{mul}x{ell}{parity}" for ell in range(lmax + 1) for parity in ("o", "e"))


def zeros(
    irreps: IntoIrreps, leading_shape: tuple = (), dtype: jnp.dtype = None, np_=jnp
) -> e3j.IrrepsArray:
    r"""Create an IrrepsArray of zeros."""
    irreps = e3j.Irreps(irreps)
    array = np_.zeros(leading_shape + (irreps.dim,), dtype=dtype)
    return e3j.IrrepsArray(irreps, array, zero_flags=(True,) * len(irreps))


def zeros_like(irreps_array: e3j.IrrepsArray) -> e3j.IrrepsArray:
    r"""Create an IrrepsArray of zeros with the same shape as another IrrepsArray."""
    np_ = infer_backend(irreps_array.array)
    return zeros(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype, np_=np_)


def ones(
    irreps: IntoIrreps, leading_shape: tuple = (), dtype: jnp.dtype = None, np_=jnp
) -> e3j.IrrepsArray:
    r"""Create an IrrepsArray of ones."""
    irreps = e3j.Irreps(irreps)
    array = np_.ones(leading_shape + (irreps.dim,), dtype=dtype)
    return e3j.IrrepsArray(irreps, array, zero_flags=(False,) * len(irreps))


def ones_like(irreps_array: e3j.IrrepsArray) -> e3j.IrrepsArray:
    r"""Create an IrrepsArray of ones with the same shape as another IrrepsArray."""
    np_ = infer_backend(irreps_array.array)
    return ones(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype, np_=np_)


def optional_import(name: str, extra: str | None = None):
    """Import a module, raising a helpful error if it's missing."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        if extra:
            hint = f" Install it with `pip install mylib[{extra}]`."
        else:
            hint = f" Install it with `pip install {name.split('.')[0]}`."

        raise ImportError(
            f"'{name}' is required for this feature but is not installed.{hint}"
        ) from e
