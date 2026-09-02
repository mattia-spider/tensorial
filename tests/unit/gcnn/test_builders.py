import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensorial
from tensorial import gcnn, nn
from tensorial.gcnn import keys

BACKBONE_KWARGS = dict(
    num_layers=3,
    mul=8,
    lmax=2,
    r_max=5.0,
    n_elements=4,
    avg_num_neighbours=20.0,
    atomic_numbers=[1, 6, 7, 8],
)


def test_build_nequip_backbone_structure():
    layers = gcnn.build_nequip_backbone(**BACKBONE_KWARGS)

    assert [type(layer).__name__ for layer in layers] == [
        "SpeciesTransform",
        "NodewiseEmbedding",
        "EdgeVectors",
        "EdgewiseEmbedding",
        "RadialBasisEdgeEmbedding",
        "NodewiseLinear",
        "NequipLayer",
        "NequipLayer",
        "NequipLayer",
    ]


def test_build_nequip_backbone_irreps():
    layers = gcnn.build_nequip_backbone(**BACKBONE_KWARGS)
    hidden = "8x0o + 8x0e + 8x1o + 8x1e + 8x2o + 8x2e"

    embed = layers[5]
    assert embed.field == keys.ATTRIBUTES
    assert embed.irreps_out == hidden

    for layer in layers[6:]:
        assert layer.irreps_out == hidden
        assert layer.num_species == 4
        assert layer.avg_num_neighbours == 20.0

    # Spherical harmonic edge embedding, and the default `r_max` threading
    assert e3j.Irreps(layers[3].attrs["edge_vectors"].irreps) == e3j.Irreps("0e + 1o + 2e")
    assert layers[4].r_max == 5.0


@pytest.mark.parametrize("num_layers,mul,lmax", [(2, 8, 2), (4, 16, 3)])
def test_build_nequip_backbone_grows(num_layers, mul, lmax):
    kwargs = BACKBONE_KWARGS | dict(num_layers=num_layers, mul=mul, lmax=lmax)
    layers = gcnn.build_nequip_backbone(**kwargs)

    nequip_layers = [layer for layer in layers if type(layer).__name__ == "NequipLayer"]
    assert len(nequip_layers) == num_layers

    expected = tensorial.utils.hidden_irreps(mul, lmax)
    assert all(layer.irreps_out == expected for layer in nequip_layers)
    assert e3j.Irreps(expected).lmax == lmax


def test_build_nequip_backbone_extra_node_encodings():
    """Extra encodings are concatenated onto the species one-hot, and stay opaque to the
    backbone."""
    extra = {
        "globals.external_magnetic_field": tensorial.tensors.SphericalHarmonic(
            "1e", normalise=False
        )
    }
    layers = gcnn.build_nequip_backbone(**BACKBONE_KWARGS, extra_node_encodings=extra)

    attrs = layers[1].attrs
    assert list(attrs) == ["species", "globals.external_magnetic_field"]
    assert attrs["species"].num_classes == 4


def test_build_nequip_backbone_rejects_no_layers():
    with pytest.raises(ValueError):
        gcnn.build_nequip_backbone(**(BACKBONE_KWARGS | dict(num_layers=0)))


def _param_shapes(params):
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    return {jax.tree_util.keystr(path): tuple(leaf.shape) for path, leaf in flat}


def test_build_nequip_backbone_matches_handwritten_stack():
    """The builder must be a drop-in for a hand-written stack: same layers, and -- because
    Sequential splices before flax binds its children -- the same auto-generated parameter names,
    so existing checkpoints stay loadable."""
    hidden = "8x0o + 8x0e + 8x1o + 8x1e + 8x2o + 8x2e"
    handwritten = [
        gcnn.atomic.SpeciesTransform(BACKBONE_KWARGS["atomic_numbers"]),
        gcnn.NodewiseEncoding(attrs={"species": tensorial.OneHot(4)}),
        gcnn.EdgeVectors(),
        gcnn.EdgewiseEncoding(
            attrs={"edge_vectors": tensorial.SphericalHarmonic("0e + 1o + 2e", normalise=True)}
        ),
        gcnn.RadialBasisEdgeEncoding(r_max=5.0),
        gcnn.NodewiseLinear(field=keys.ATTRIBUTES, irreps_out=hidden),
        *(
            gcnn.NequipLayer(num_species=4, irreps_out=hidden, avg_num_neighbours=20.0)
            for _ in range(3)
        ),
    ]
    readout = gcnn.NodewiseLinear(irreps_out="1x0e + 1x1e + 1x2e", out_field="predicted")

    old = nn.Sequential([*handwritten, readout])
    new = nn.Sequential([gcnn.build_nequip_backbone(**BACKBONE_KWARGS), readout])

    rng = np.random.default_rng(0)
    graph = gcnn.graph_from_points(
        rng.random((12, 3)) * 4.0,
        r_max=5.0,
        nodes={
            gcnn.atomic.keys.ATOMIC_NUMBERS: jnp.asarray(
                rng.choice(BACKBONE_KWARGS["atomic_numbers"], size=12)
            )
        },
    )
    graph = jax.tree.map(jnp.asarray, graph)

    key = jax.random.PRNGKey(0)
    params_old, params_new = old.init(key, graph), new.init(key, graph)
    assert _param_shapes(params_old) == _param_shapes(params_new)

    out_old = old.apply(params_old, graph).nodes["predicted"]
    out_new = new.apply(params_new, graph).nodes["predicted"]
    np.testing.assert_allclose(
        tensorial.as_array(out_old), tensorial.as_array(out_new), rtol=1e-6, atol=1e-6
    )
