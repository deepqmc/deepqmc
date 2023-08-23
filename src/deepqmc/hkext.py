import haiku as hk
import jax.numpy as jnp
from haiku.initializers import VarianceScaling
from jax import tree_util
from jax.nn import softplus


def ssp(x):
    r"""
    Compute the shifted softplus activation function.

    Computes the elementwise function
    :math:`\text{softplus}(x)=\log(1+\text{e}^x)+\log\frac{1}{2}`
    """
    return softplus(x) + jnp.log(0.5)


class MLP(hk.Module):
    r"""
    Represent a multilayer perceptron.

    Args:
        in_dim (int): the input dimension.
        out_dim (int): the output dimension.
        residual (bool): whether to include a residual connection
        name (str): optional, the name of the network.
        hidden_layers (tuple): optional, either ('log', :math:`N_\text{layers}`),
            in which case the network will have :math:`N_\text{layers}` layers
            with logarithmically changing widths, or a tuple of ints specifying
            the width of each layer.
        bias (str): optional, specifies which layers should have a bias term.
            Possible values are

            - :data:`True`: all layers will have a bias term
            - :data:`False`: no layers will have a bias term
            - ``'not_last'``: all but the last layer will have a bias term
        last_linear (bool): optional, if :data:`True` the activation function
            is not applied to the activation of the last layer.
        activation (Callable): optional, the activation function.
        w_init (str or Callable): optional, specifies the initialization of the
            linear weights. Possible string values are:

            - ``'default'``: the default haiku initialization method is used.
            - ``'deeperwin'``: the initialization method of the :class:`deeperwin`
                package is used.
    """

    def __init__(
        self,
        out_dim,
        name=None,
        *,
        hidden_layers,
        bias,
        last_linear,
        activation,
        init,
    ):
        assert bias in (True, False, 'not_last')
        super().__init__(name=name)
        self.activation = activation
        self.last_linear = last_linear
        self.bias = bias
        self.out_dim = out_dim
        if isinstance(init, str):
            self.w_init = {
                'deeperwin': VarianceScaling(1.0, 'fan_avg', 'uniform'),
                'default': VarianceScaling(1.0, 'fan_in', 'truncated_normal'),
                'ferminet': VarianceScaling(1.0, 'fan_in', 'normal'),
            }[init]
            self.b_init = {
                'deeperwin': lambda s, d: jnp.zeros(shape=s, dtype=d),
                'default': lambda s, d: jnp.zeros(shape=s, dtype=d),
                'ferminet': VarianceScaling(1.0, 'fan_out', 'normal'),
            }[init]
        else:
            self.w_init = init
            self.b_init = init
        self.hidden_layers = hidden_layers or []

    def __call__(self, inputs):
        if len(self.hidden_layers) == 2 and self.hidden_layers[0] == 'log':
            n_hidden = self.hidden_layers[1]
            qs = [k / n_hidden for k in range(1, n_hidden + 1)]
            dims = [round(inputs.shape[-1] ** (1 - q) * self.out_dim**q) for q in qs]
        else:
            dims = [*self.hidden_layers, self.out_dim]
        n_layers = len(dims)
        layers = []
        for idx, dim in enumerate(dims):
            with_bias = self.bias is True or (
                self.bias == 'not_last' and idx < (n_layers - 1)
            )
            layers.append(
                hk.Linear(
                    output_size=dim,
                    with_bias=with_bias,
                    name='linear_%d' % idx,
                    w_init=self.w_init,
                    b_init=self.b_init,
                )
            )

        out = inputs
        for i, layer in enumerate(layers):
            out = layer(out)
            if i < (n_layers - 1) or not self.last_linear:
                out = self.activation(out)
        return out


class ResidualConnection:
    r"""
    Represent a residual connection between pytrees.

    The residual connection is only added if :data:`inp` and :data:`update`
    have the same shape.

    Args:
        - normalize (bool): if :data:`True` the sum of :data:`inp` and :data:`update`
            is normalized with :data:`sqrt(2)`.
    """

    def __init__(self, *, normalize):
        self.normalize = normalize

    def __call__(self, inp, update):
        def leaf_residual(x, y):
            if x.shape != y.shape:
                return y
            z = x + y
            return z / jnp.sqrt(2) if self.normalize else z

        return tree_util.tree_map(leaf_residual, inp, update)


class SumPool:
    r"""Represent a global sum pooling operation."""

    def __init__(self, out_dim, name=None):
        assert out_dim == 1

    def __call__(self, x):
        return tree_util.tree_map(lambda leaf: leaf.sum(axis=-1, keepdims=True), x)


class Identity:
    r"""Represent the identity operation."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x
