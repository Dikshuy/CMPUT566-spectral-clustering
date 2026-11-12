import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from spectral.allo import Batch, Learner
from spectral.config import Config
from spectral.data import make_moons
from spectral.exact import compare, exact_eigs
from spectral.graph import knn_graph


class GraphData(NamedTuple):
    x: jax.Array        # (n, dim) features
    src: jax.Array      # (E,) ordered edges
    dst: jax.Array
    p_edge: jax.Array   # (E,) W_ij / vol


def sample_batch(data: GraphData, key: jax.Array, batch_size: int) -> Batch:
    """edges proportional to W_ij; constraint points uniformly"""
    k_edge, k1, k2 = jax.random.split(key, 3)
    e = jax.random.choice(k_edge, data.src.shape[0], (batch_size,), p=data.p_edge)
    c1, c2 = (jax.random.randint(k, (batch_size,), 0, data.x.shape[0]) for k in (k1, k2))
    return Batch(data.x[data.src[e]], data.x[data.dst[e]], data.x[c1], data.x[c2])


def make_chunk_fn(graphdef, batch_size: int, length: int):
    """jit(vmap over seeds); returns the new state and per-seed mean metrics"""
    def seed_chunk(state, data, key, start):
        def body(state, step):
            model = nnx.merge(graphdef, state)
            metrics = model.update(sample_batch(data, jax.random.fold_in(key, step), batch_size))
            return nnx.state(model), metrics

        state, metrics = jax.lax.scan(body, state, start + jnp.arange(length))
        return state, jax.tree.map(lambda m: m.mean(0), metrics)

    return jax.jit(jax.vmap(seed_chunk, in_axes=(0, None, 0, None)), donate_argnums=0)


def run(cfg: Config) -> dict:
    x, y = make_moons(cfg.data)
    graph = knn_graph(x, cfg.graph)
    lam, V = exact_eigs(graph, cfg.allo.d)
    print(f"[graph] n={graph.n} edges={graph.W.nnz} components={graph.n_components()}")
    print(f"[exact] lambda = {np.array2string(lam, precision=4)}")

    src, dst, w = graph.edges()
    data = GraphData(jnp.asarray(x), jnp.asarray(src), jnp.asarray(dst), jnp.asarray(w / w.sum(), jnp.float32))

    build = lambda key: Learner(cfg.allo, x.shape[1], x.mean(0), x.std(0), cfg.run.steps, nnx.Rngs(key))
    graphdef, _ = nnx.split(nnx.eval_shape(lambda: build(jax.random.key(0))))
    keys = jnp.stack([jax.random.key(s) for s in cfg.run.seeds])
    state = jax.jit(jax.vmap(lambda k: nnx.split(build(k))[1]))(keys)
    chunk = make_chunk_fn(graphdef, cfg.run.batch_size, cfg.run.log_every)
    embed = jax.jit(jax.vmap(lambda s: (lambda m: (m.allo(data.x), m.allo.eigenvalues()))(nnx.merge(graphdef, s))))

    L = graph.laplacian()
    history, t = [], time.perf_counter()
    print(f"{'step':>7} {'graph':>8} {'orth':>7} {'barrier':>7} {'cos_mean':>8} {'subspace':>8} {'eig_err':>8} {'ray_err':>8} {'time':>6}")
    for start in range(0, cfg.run.steps, cfg.run.log_every):
        state, metrics = chunk(state, data, keys, jnp.int32(start))
        U, mu_hat = map(np.asarray, embed(state))
        lam_hat = mu_hat * graph.vol / graph.n      # ALLO's duals learn mu = lambda * n / vol
        evals = [compare(U[s], V, lam_hat[s], lam, L) for s in range(len(cfg.run.seeds))]
        row = {"step": start + cfg.run.log_every, **{k: float(v.mean()) for k, v in metrics.items()},
               **{k: np.mean([e[k] for e in evals], 0) for k in evals[0]}}
        history.append(row)
        print(f"{row['step']:>7} {row['graph_loss']:>8.4f} {row['orth_err']:>7.4f} {row['barrier']:>7.3f} "
              f"{row['cosine'].mean():>8.3f} {row['subspace']:>8.4f} {row['eig_abs_err'].mean():>8.2e} "
              f"{row['rayleigh_abs_err'].mean():>8.2e} {time.perf_counter() - t:>5.0f}s")

    print(f"[final] cosine per eigenvector (mean over seeds) = {np.array2string(history[-1]['cosine'], precision=3)}")
    print(f"[final] lambda     (exact)  = {np.array2string(lam, precision=4)}")
    print(f"[final] lambda_hat (duals)  = {np.array2string(lam_hat[0], precision=4)}  (seed 0)")
    print(f"[final] rayleigh   (U)      = {np.array2string(evals[0]['rayleigh'], precision=4)}  (seed 0)")
    return {"x": x, "y": y, "U": U, "lam_hat": lam_hat, "V": V, "lam": lam, "history": history}
