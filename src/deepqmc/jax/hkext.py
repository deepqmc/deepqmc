import haiku as hk
import jax.numpy as jnp
from haiku.initializers import VarianceScaling
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
        name (str): optional, the name of the network.
        w_init (str or Callable): optional, specifies the initialization of the
            linear weights. Possible string values are:

            - ``'default'``: the default haiku initialization method is used.
            - ``'deeperwin'``: the initialization method of the :class:`deeperwin`
                package is used.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers=None,
        bias=True,
        last_linear=False,
        activation=ssp,
        name=None,
        w_init='default',
    ):
        assert bias in (True, False, 'not_last')
        super().__init__(name=name)
        self.activation = activation
        self.last_linear = last_linear
        if isinstance(w_init, str):
            w_init = {
                'deeperwin': VarianceScaling(1.0, 'fan_avg', 'uniform'),
                'default': VarianceScaling(1.0, 'fan_in', 'truncated_normal'),
            }[w_init]
        hidden_layers = hidden_layers or []
        if len(hidden_layers) == 2 and hidden_layers[0] == 'log':
            n_hidden = hidden_layers[1]
            qs = [k / n_hidden for k in range(1, n_hidden + 1)]
            dims = [round(in_dim ** (1 - q) * out_dim**q) for q in qs]
        else:
            dims = [*hidden_layers, out_dim]
        self.layers = []
        for idx, dim in enumerate(dims):
            with_bias = bias is True or (bias == 'not_last' and idx < (len(dims) - 1))
            self.layers.append(
                hk.Linear(
                    output_size=dim,
                    with_bias=with_bias,
                    name='linear_%d' % idx,
                    w_init=w_init,
                )
            )

    def __call__(self, inputs):
        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (len(self.layers) - 1) or not self.last_linear:
                out = self.activation(out)
        return out
