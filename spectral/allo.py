from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from spectral.config import ALLOConfig
from spectral.networks import MLP


class Batch(NamedTuple):
    src: jax.Array      # (B, dim) edge endpoints, pairs drawn proportionally to W_ij
    dst: jax.Array
    c1: jax.Array       # (B, dim) two independent uniform draws of points for the constraint
    c2: jax.Array


class ALLO(nnx.Module):
    """u(x), with the duals and barrier of its orthonormality constraint"""
    def __init__(self, cfg: ALLOConfig, x_dim: int, mean, std, rngs: nnx.Rngs):
        self.mlp = MLP(x_dim, cfg.d, cfg.net, rngs, mean, std)
        self.duals = nnx.Param(jnp.tril(jnp.full((cfg.d, cfg.d), cfg.duals_init)))
        self.barrier = nnx.Param(jnp.asarray(cfg.barrier_init))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.mlp(x)

    def eigenvalues(self) -> jax.Array:
        """mu_i = -beta_ii / 2"""
        return -0.5 * jnp.diag(self.duals[...])


def _violation(u: jax.Array) -> jax.Array:
    """tril(E[u u^T] - I) over the batch, with the gradient through one factor"""
    gram = jnp.einsum("bi,bj->ij", u, jax.lax.stop_gradient(u)) / u.shape[0]
    return jnp.tril(gram - jnp.eye(u.shape[1]))


class Learner(nnx.Module):
    def __init__(self, cfg: ALLOConfig, x_dim: int, mean, std, rngs: nnx.Rngs):
        self.cfg = cfg
        self.allo = ALLO(cfg, x_dim, mean, std, rngs)
        self.opt = nnx.Optimizer(self.allo, optax.adam(cfg.lr), wrt=nnx.Param)

    def update(self, batch: Batch) -> dict:
        cfg = self.cfg

        def loss_fn(allo):
            # graph loss
            graph = 0.5 * ((allo(batch.src) - allo(batch.dst)) ** 2).sum(-1).mean()
            err1, err2 = _violation(allo(batch.c1)), _violation(allo(batch.c2))
            err, sq_err = 0.5 * (err1 + err2), err1 * err2

            # duals and barrier losses
            duals, barrier = allo.duals[...], allo.barrier[...]
            dual_primal = (jax.lax.stop_gradient(duals) * err).sum()
            dual_ascent = -cfg.duals_step_size * (duals * jax.lax.stop_gradient(err)).sum()
            barrier_primal = jax.lax.stop_gradient(barrier) * sq_err.sum()
            barrier_ascent = -barrier * jax.lax.stop_gradient(jnp.clip(sq_err, 0.0, None).mean())

            # total loss
            loss = graph + dual_primal + dual_ascent + barrier_primal + barrier_ascent
            return loss, {"graph_loss": graph, "barrier": barrier}

        # update parameters
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.allo)
        self.opt.update(self.allo, grads)
        self.allo.duals[...] = jnp.clip(self.allo.duals[...], cfg.duals_min, cfg.duals_max)
        self.allo.barrier[...] = jnp.clip(self.allo.barrier[...], cfg.barrier_min, cfg.barrier_max)
        return {"allo_loss": loss, **metrics}
