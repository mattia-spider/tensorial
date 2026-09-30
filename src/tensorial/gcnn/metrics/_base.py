from collections.abc import Sequence
import math
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Optional, TypeVar

import beartype
import jax.numpy as jnp
import jax.typing
import jaxtyping as jt
from jaxtyping import Array, Int
import jraph
from pytray import tree
import reax
from typing_extensions import override

from .. import _tree, keys
from ... import nn_utils

if TYPE_CHECKING:
    from tensorial import gcnn

OutT = TypeVar("OutT")

__all__ = (
    "GraphMetric",
    "graph_metric",
    "species_metric",
    "species_metrics",
    "AvgNumNeighboursByType",
)


@jt.jaxtyped(typechecker=beartype.beartype)
def graph_metric(
    metric: str | reax.Metric | type[reax.Metric],
    predictions: "gcnn.typing.TreePathLike",
    targets: "Optional[gcnn.TreePathLike]" = None,
    mask: "Optional[gcnn.TreePathLike | Literal['auto']]" = "auto",
    normalise_by: "Optional[gcnn.TreePathLike]" = None,
) -> "GraphMetric":
    predictions_from = _tree.path_from_str(predictions)
    targets_from = _tree.path_to_str(targets) if targets is not None else None
    mask_from = _tree.path_to_str(mask) if mask is not None else None
    norm_by = _tree.path_to_str(normalise_by) if normalise_by is not None else None

    class _GraphMetric(GraphMetric):
        parent = reax.metrics.get(metric)
        pred_key = predictions_from
        target_key = targets_from
        mask_key = mask_from
        normalise_by = norm_by

    return _GraphMetric()


# Written into `nodes` on the fly so that `GraphMetric` can find it by path, the same way it
# finds the padding mask
_SPECIES_MASK: Final[str] = "_species_mask"


def _with_species_mask(
    graph: jraph.GraphsTuple, species_path, species_index: int
) -> jraph.GraphsTuple:
    """Return ``graph`` with a boolean node mask selecting only ``species_index``."""
    species = jnp.asarray(tree.get_by_path(graph._asdict(), species_path)).reshape(-1)
    mask = species == species_index

    # Respect any padding mask that is already there, otherwise padded nodes get counted
    padding = graph.nodes.get("mask")
    if padding is not None:
        mask = jnp.logical_and(mask, jnp.asarray(padding).reshape(-1))

    nodes = dict(graph.nodes)
    nodes[_SPECIES_MASK] = mask
    return graph._replace(nodes=nodes)


@jt.jaxtyped(typechecker=beartype.beartype)
def species_metric(
    metric: str | reax.Metric | type[reax.Metric],
    predictions: "gcnn.typing.TreePathLike",
    species: int,
    targets: "Optional[gcnn.TreePathLike]" = None,
    atomic_numbers: Sequence[int] | None = None,
    species_field: "Optional[gcnn.TreePathLike]" = None,
    normalise_by: "Optional[gcnn.TreePathLike]" = None,
) -> "GraphMetric":
    """Evaluate ``metric`` over the nodes of a single species.

    Behaves like :func:`graph_metric`, but restricts the reduction to the nodes whose species
    matches, which is what separates e.g. a per element RMSE from the aggregate one.

    :param species: the species to select.  Together with ``atomic_numbers`` this is an atomic
        number (1 for hydrogen), otherwise it is the index used by
        :class:`~tensorial.gcnn.atomic.SpeciesTransform`.
    :param atomic_numbers: the type map the model was built with, used to turn an atomic number
        into a species index.
    """
    if atomic_numbers is not None:
        atomic_numbers = list(atomic_numbers)
        if species not in atomic_numbers:
            raise ValueError(f"species {species} is not in the type map {atomic_numbers}")
        species_index = atomic_numbers.index(species)
    else:
        species_index = species

    predictions_from = _tree.path_from_str(predictions)
    targets_from = _tree.path_to_str(targets) if targets is not None else None
    norm_by = _tree.path_to_str(normalise_by) if normalise_by is not None else None
    species_from = _tree.path_from_str(
        species_field if species_field is not None else f"nodes.{keys.SPECIES}"
    )

    class _SpeciesMetric(GraphMetric):
        parent = reax.metrics.get(metric)
        pred_key = predictions_from
        target_key = targets_from
        mask_key = _tree.path_to_str(f"nodes.{_SPECIES_MASK}")
        normalise_by = norm_by

        @override
        def create(self, predictions, targets=None):  # pylint: disable=arguments-differ
            return super().create(
                _with_species_mask(predictions, species_from, species_index), targets
            )

    return _SpeciesMetric()


def _species_label(atomic_number: int) -> str:
    try:
        from ase.data import chemical_symbols  # pylint: disable=import-outside-toplevel
    except ImportError:
        return f"Z{atomic_number}"

    if 0 < atomic_number < len(chemical_symbols):
        return chemical_symbols[atomic_number]
    return f"Z{atomic_number}"


@jt.jaxtyped(typechecker=beartype.beartype)
def species_metrics(
    metric: str | reax.Metric | type[reax.Metric],
    predictions: "gcnn.typing.TreePathLike",
    atomic_numbers: Sequence[int] | jax.Array,
    targets: "Optional[gcnn.TreePathLike]" = None,
    species: Sequence[int] | None = None,
    species_field: "Optional[gcnn.TreePathLike]" = None,
    normalise_by: "Optional[gcnn.TreePathLike]" = None,
) -> dict[str, "GraphMetric"]:
    """Build a :func:`species_metric` for every species in the type map.

    Meant to be fed the type map gathered from the data (``${from_data.atomic_numbers}``), so
    the per element metrics follow the dataset instead of being listed by hand.  The result is
    keyed by element symbol, and :class:`~tensorial.reaxkit.ReaxModule` flattens it into
    ``<name>_<symbol>`` entries, e.g. ``nmr_rmse: {H: ..., C: ...}`` is logged as
    ``nmr_rmse_H``, ``nmr_rmse_C``.

    :param atomic_numbers: the type map the model was built with.
    :param species: restrict to these atomic numbers, all of ``atomic_numbers`` if ``None``.
    """
    atomic_numbers = [int(z) for z in jnp.asarray(atomic_numbers).reshape(-1).tolist()]
    selected = atomic_numbers if species is None else [int(z) for z in species]

    return {
        _species_label(z): species_metric(
            metric,
            predictions,
            species=z,
            targets=targets,
            atomic_numbers=atomic_numbers,
            species_field=species_field,
            normalise_by=normalise_by,
        )
        for z in selected
    }


def mdiv(
    num: jax.typing.ArrayLike, denom: jax.typing.ArrayLike, where: jax.typing.ArrayLike = None
):
    """Divide that supports supplying a mask, where `False` values will just return the numerator"""
    # Use prod here because `IrrepsArray` doesn't have `.size`
    if math.prod(num.shape) != math.prod(denom.shape):
        raise ValueError(
            "Sizes of numerator and denominator must match, got {num.shape} and {denom.shape}"
        )
    if where is not None:
        where = reax.metrics.utils.prepare_mask(denom, where)
        denom = jnp.where(where, denom, 1.0)

    return num / denom.reshape(num.shape)


class GraphMetric(reax.Metric):
    parent: ClassVar[reax.Metric]
    pred_key: "ClassVar[gcnn.typing.TreePathLike]"
    target_key: "ClassVar[Optional[gcnn.typing.TreePathLike]]" = None
    mask_key: "ClassVar[Optional[gcnn.typing.TreePathLike]]" = "auto"
    normalise_by: "ClassVar[Optional[gcnn.typing.TreePathLike]]" = None

    _state: reax.Metric[OutT] | None

    def __init__(self, state: reax.Metric[OutT] | None = None):
        super().__init__()
        self._state = state

    @property
    def metric(self) -> reax.Metric[OutT] | None:
        return self._state

    @property
    def is_empty(self) -> bool:
        return self._state is None

    @override
    def create(
        # pylint: disable=arguments-differ
        self,
        predictions: jraph.GraphsTuple,
        targets: jraph.GraphsTuple | None = None,
    ) -> "GraphMetric":
        if targets is None:
            # In this case, the user is typically using a different key in the same graph
            targets = predictions

        pred = _tree.get(predictions, self.pred_key)

        mask = None
        if self.mask_key is not None:
            if self.mask_key == "auto":
                pred_key = _tree.path_from_str(self.pred_key)
                mask_key = pred_key[:-1] + ("mask",)
            else:
                mask_key = self.mask_key

            try:
                mask = _tree.get(predictions, mask_key)
            except KeyError:
                mask = None

        if self.normalise_by is not None:
            pred = mdiv(pred, _tree.get(predictions, self.normalise_by), where=mask)

        args = [pred]
        # If there is a target field, add that to the argument list
        if self.target_key:
            targ = _tree.get(targets, self.target_key)
            if self.normalise_by is not None:
                targ = mdiv(targ, _tree.get(targets, self.normalise_by), where=mask)

            args.append(targ)

        kwargs = {} if mask is None else {"mask": mask}

        return type(self)(self.parent.create(*args, **kwargs))

    @override
    def merge(self, other: "GraphMetric") -> "GraphMetric":
        if other.is_empty:
            return self
        if self.is_empty:
            return other

        return type(self)(self._state.merge(other._state))  # pylint: disable=protected-access

    @override
    def compute(self) -> OutT:
        if self.is_empty:
            raise RuntimeError("Cannot compute, metric is empty")

        return self._state.compute()

    @override
    def reduce(self, axis: int = 0) -> "reax.Metric[OutT]":
        if self.is_empty:
            raise RuntimeError("Cannot compute, metric is empty")

        return type(self)(self._state.reduce(axis=axis))


class AvgNumNeighboursByType(reax.Metric[dict[int, jax.Array]]):
    """Get the average number of node neighbours grouped by node type where the type is an integer
    found in G.nodes[type_field].
    """

    Averages = list[reax.metrics.Average]
    _type_field: str
    _node_types: Int[Array, "n_types"]
    _state: Averages | None

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        node_types: Sequence[int] | Int[Array, "n_types"],
        type_field: str = "type_id",
        state: Averages | None = None,
    ):
        """
        Initializes the AvgNumNeighboursByType class with node types, type field, and optional
        state.

        Args:
            node_types: Sequence of integers representing the types of nodes or an array with shape
                (n_types,).
            type_field: String indicating the field name used to identify node types, defaults to
                "type_id".
            state: Optional Averages object containing precomputed state information.

        Raises:
            ValueError: If node_types is empty or contains invalid values.
            TypeError: If type_field is not a string or state is not of type Averages or None.
        """
        self._node_types = jnp.asarray(node_types)
        self._type_field = type_field
        self._state: AvgNumNeighboursByType.Averages | None = state

    @property
    def is_empty(self) -> bool:
        return self._state is None

    def empty(self) -> "AvgNumNeighboursByType":
        if self.is_empty:
            return self

        return AvgNumNeighboursByType(self._node_types)

    def merge(self, other: "AvgNumNeighboursByType") -> "AvgNumNeighboursByType":
        # pylint: disable=protected-access
        if self._node_types.shape != other._node_types.shape or not bool(
            jnp.all(self._node_types == other._node_types)
        ):
            raise ValueError(
                f"Type maps must match, got {self._node_types} and {other._node_types}"
            )

        if other.is_empty:  # pylint: disable=protected-access
            return self
        if self.is_empty:
            return other

        return AvgNumNeighboursByType(
            node_types=self._node_types,
            state=[
                avg.merge(other_avg)
                for avg, other_avg in zip(
                    self._state, other._state  # pylint: disable=protected-access
                )
            ],
        )

    def create(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "AvgNumNeighboursByType":
        state = self._calc_averages(graphs)  # pylint: disable=not-callable
        return AvgNumNeighboursByType(
            node_types=self._node_types, type_field=self._type_field, state=state
        )

    def update(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "AvgNumNeighboursByType":
        if self.is_empty:
            return self.create(graphs)

        # Create the updated state
        state = [
            avg.merge(other_avg) for avg, other_avg in zip(self._state, self._calc_averages(graphs))
        ]
        return AvgNumNeighboursByType(node_types=self._node_types, state=state)

    def compute(self) -> dict[int, jax.Array]:
        if self.is_empty:
            raise RuntimeError("Nothing to compute, metric is empty!")

        return {
            type_id: avg.compute() for type_id, avg in zip(self._node_types.tolist(), self._state)
        }

    @jt.jaxtyped(typechecker=beartype.beartype)
    def _calc_averages(self, graphs: jraph.GraphsTuple, *_) -> Averages:
        graph_dict = graphs._asdict()

        types = tree.get_by_path(graph_dict, ("nodes", self._type_field))[:, 0]
        # Transform the type numbers from whatever they are to 0, 1, 2....
        types = nn_utils.vwhere(types, self._node_types)

        counts = jnp.bincount(graphs.senders, length=jnp.sum(graphs.n_node).item())
        mask = reax.metrics.utils.prepare_mask(counts, graphs.nodes.get(keys.MASK))
        mask = mask if mask is not None else True

        num_classes = len(self._node_types)
        return [
            reax.metrics.Average.create(counts, mask & (types == idx)) for idx in range(num_classes)
        ]
