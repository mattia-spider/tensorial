"""Factories that assemble commonly used stacks of graph layers.

These are deliberately task agnostic: they build a feature extractor and stop there.  Whatever
reads those features out (a readout head, a decoding to Cartesian tensors, ...) belongs to the
application and should be composed alongside the backbone, not baked into it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from flax import linen

from . import _edgewise, _nequip, _nodewise, atomic, keys
from ..typing import IntoIrreps
from ..utils import hidden_irreps

__all__ = "build_nequip_backbone", "hidden_irreps"


def build_nequip_backbone(  # pylint: disable=too-many-arguments
    *,
    num_layers: int,
    mul: int,
    lmax: int,
    r_max: float,
    n_elements: int,
    avg_num_neighbours: float,
    atomic_numbers: Sequence[int],
    sh_irreps: IntoIrreps = "0e + 1o + 2e",
    extra_node_encodings: Mapping[str, Any] = None,
) -> list[linen.Module]:
    """Build a NequIP-style feature extractor as a flat list of layers.

    The returned layers embed the species (plus anything in ``extra_node_encodings``), build the
    edge geometry, and run ``num_layers`` interaction layers.  The node features left in
    :attr:`tensorial.gcnn.keys.FEATURES` have irreps ``hidden_irreps(mul, lmax)``.

    There is no reduction, readout or decoding here.  Drop the returned list straight into a
    :class:`tensorial.nn.Sequential` (it splices nested lists) and follow it with whatever head
    the task needs.

    Args:
        num_layers: the number of NequIP interaction layers
        mul: the multiplicity (number of channels) of the hidden features
        lmax: the maximum rotation order of the hidden features
        r_max: the cutoff radius used by the radial basis
        n_elements: the number of distinct species in the dataset
        avg_num_neighbours: average number of neighbours, used to normalise the message passing
        atomic_numbers: the atomic numbers present in the dataset, used to map them onto
            contiguous species indices
        sh_irreps: the irreps of the spherical harmonic edge embedding
        extra_node_encodings: additional node attributes to concatenate onto the species
            encoding, as a mapping of field path (e.g. ``"globals.external_magnetic_field"``) to
            the attribute describing how to encode it.  Lets a caller inject extra inputs without
            the backbone needing to know what they are.

    Returns:
        the backbone layers, in order
    """
    # Imported here to keep `tensorial.gcnn` importable before `tensorial.tensors` is ready
    from .. import tensors  # pylint: disable=import-outside-toplevel,cyclic-import

    if num_layers < 1:
        raise ValueError(f"'num_layers' must be at least 1, got {num_layers}")

    hidden = hidden_irreps(mul, lmax)

    node_attrs: dict[str, Any] = {"species": tensors.OneHot(n_elements)}
    node_attrs.update(extra_node_encodings or {})

    layers: list[linen.Module] = [
        atomic.SpeciesTransform(atomic_numbers),
        _nodewise.NodewiseEncoding(attrs=node_attrs),
        _edgewise.EdgeVectors(),
        _edgewise.EdgewiseEncoding(
            attrs={"edge_vectors": tensors.SphericalHarmonic(sh_irreps, normalise=True)}
        ),
        _edgewise.RadialBasisEdgeEncoding(r_max=r_max),
        _nodewise.NodewiseLinear(field=keys.ATTRIBUTES, irreps_out=hidden),
    ]

    # `NequipLayer` infers its input irreps from the incoming node features at call time, so the
    # stack only ever needs to be told what to produce
    layers.extend(
        _nequip.NequipLayer(
            num_species=n_elements,
            irreps_out=hidden,
            avg_num_neighbours=avg_num_neighbours,
        )
        for _ in range(num_layers)
    )

    return layers
