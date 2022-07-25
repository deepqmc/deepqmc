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
        hidden_layers=[],
        last_bias=True,
        last_linear=False,
        activation=SSP,
        name=None,
    ):
        super().__init__(name=name)
        self.activation = activation
        self.last_linear = last_linear
        if len(hidden_layers) == 2 and hidden_layers[0] == 'log':
            n_hidden = hidden_layers[1]
            qs = [k / n_hidden for k in range(1, n_hidden + 1)]
            dims = [int(jnp.round(in_dim ** (1 - q) * out_dim**q)) for q in qs]
        else:
            dims = [*hidden_layers, out_dim]
        layers = []
        for index, dim in enumerate(dims):
            with_bias = index < (len(dims) - 1) or last_bias
            layers.append(
                hk.Linear(
                    output_size=dim, with_bias=with_bias, name="linear_%d" % index
                )
            )
        self.layers = layers

    def __call__(self, inputs):
        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (len(self.layers) - 1) or not self.last_linear:
                out = self.activation(out)
        return out
