import math

import jax
import jax.numpy as jnp
from flax import nnx

from spectral.config import NetConfig

ZEROS = nnx.initializers.zeros_init()


def linear(d_in: int, d_out: int, rngs: nnx.Rngs, gain: float = 1.0) -> nnx.Linear:
    """linear layer with orthogonal weight initialisation"""
    return nnx.Linear(d_in, d_out, kernel_init=nnx.initializers.orthogonal(scale=gain), bias_init=ZEROS, rngs=rngs)


class MLP(nnx.Module):
    """standardise -> (Linear + relu)^{hid_layers} -> Linear"""
    def __init__(self, d_in: int, d_out: int, cfg: NetConfig, rngs: nnx.Rngs, mean, std):
        self.mean, self.std = nnx.Variable(jnp.asarray(mean)), nnx.Variable(jnp.asarray(std))
        dims = [d_in] + [cfg.hid_dim] * cfg.hid_layers
        self.hidden = nnx.List([linear(a, b, rngs, math.sqrt(2.0)) for a, b in zip(dims[:-1], dims[1:])])
        self.out = linear(cfg.hid_dim, d_out, rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = (x - self.mean[...]) / self.std[...]
        for layer in self.hidden:
            h = nnx.relu(layer(h))
        return self.out(h)
