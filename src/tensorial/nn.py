from collections.abc import Sequence
import functools

from flax import linen


class Sequential(linen.Module):
    """Applies a sequential chain of modules just like :class:`flax.linen.Sequential` _except_ that
    flax's version will expand any tuples that it receives when calling the next layer.  This
    doesn't play nice with types that subclass `tuple`, for example, :class:`jraph.GraphsTuple`,
    because the layers expect to get a `GraphsTuple`, not the individual values that make it up.

    Our behaviour is the same as :class:`flax.linen.Sequential` if we get a `tuple`, but any
    subclasses thereof are kept intact when calling the next layer.

    Entries that are themselves plain lists (or tuples) of layers are spliced into the chain
    rather than being treated as a single layer.  This lets a factory function that builds a
    group of layers (e.g. :func:`tensorial.gcnn.build_nequip_backbone`) be dropped straight into
    ``layers`` alongside hand-written ones.
    """

    layers: Sequence[linen.Module | functools.partial]

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._layers: list[linen.Module] = _layers(self.layers)

    def __post_init__(self):
        if not isinstance(self.layers, Sequence) or isinstance(self.layers, str):
            raise ValueError(f"'layers' must be a sequence, got '{type(self.layers).__name__}'.")

        # Splice before flax gets to bind the children: it derives the auto-generated parameter
        # names from the structure of this attribute, so a chain built from nested lists gets the
        # same names as the equivalent flat one
        object.__setattr__(self, "layers", tuple(_flatten(self.layers)))

        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")
        super().__post_init__()

    @linen.compact
    def __call__(self, *args, **kwargs):
        outputs = self._layers[0](*args, **kwargs)
        for layer in self._layers[1:]:
            if isinstance(outputs, dict):
                outputs = layer(**outputs)
            elif type(outputs) is tuple:  # pylint: disable=unidiomatic-typecheck
                outputs = layer(**outputs)
            else:
                outputs = layer(outputs)
        return outputs


def _flatten(layers: Sequence) -> list:
    """Splice any nested plain lists/tuples of layers into a single flat sequence.

    Nested :class:`Sequential` instances are deliberately left alone: they are modules in their
    own right and flattening them would change the parameter scoping of existing models.
    """
    flat = []
    for layer in layers:
        if isinstance(layer, (list, tuple)) and not isinstance(layer, linen.Module):
            flat.extend(_flatten(layer))
        else:
            flat.append(layer)

    return flat


def _layers(layers: Sequence[linen.Module | functools.partial]) -> list[linen.Module]:
    """Create the model from the configuration object"""
    new_layers = []
    for layer in _flatten(layers):
        if isinstance(layer, functools.partial):
            # We've reached a module that is partly constructed.  This indicates that it's a
            # module that wraps a function i.e. f(g(x)), typically because it needs access to
            # g(x) (for example to calculate gradients). So, we build what we've found so far,
            # and pass it to the module
            if len(new_layers) == 0:
                raise ValueError(
                    f"Got a partial module, but have no previous modules to pass to it: {layer}"
                )

            if len(new_layers) == 1:
                nested = new_layers[0]
            else:
                nested = Sequential(new_layers)

            layer = layer(nested)
            if not isinstance(layer, linen.Module):
                raise ValueError(
                    f"Calling partial module {type(layer).__name__}() did not resolve to a "
                    f"linen.Module instance"
                )

            new_layers = [layer]
        else:
            new_layers.append(layer)

    if len(new_layers) == 1:
        # Special case to avoid needlessly wrapping a single module
        return [new_layers[0]]

    return new_layers
