import haiku as hk
import jax.numpy as jnp
from jax.nn import softplus


def SSP(x):
    return softplus(x) + jnp.log(0.5)


class MLP(hk.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers=None,
        bias='all',
        last_linear=False,
        activation=SSP,
        name=None,
    ):
        super().__init__(name=name)
        self.activation = activation
        self.last_linear = last_linear
        hidden_layers = hidden_layers or []
        if len(hidden_layers) == 2 and hidden_layers[0] == 'log':
            n_hidden = hidden_layers[1]
            qs = [k / n_hidden for k in range(1, n_hidden + 1)]
            dims = [round(in_dim ** (1 - q) * out_dim**q) for q in qs]
        else:
            dims = [*hidden_layers, out_dim]
        layers = []
        for idx, dim in enumerate(dims):
            with_bias = bias == 'all' or (bias == 'not_last' and idx < (len(dims) - 1))
            layers.append(
                hk.Linear(output_size=dim, with_bias=with_bias, name='linear_%d' % idx)
            )
        self.layers = layers

    def __call__(self, inputs):
        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (len(self.layers) - 1) or not self.last_linear:
                out = self.activation(out)
        return out
