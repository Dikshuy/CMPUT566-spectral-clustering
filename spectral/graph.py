import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from spectral.config import GraphConfig


@dataclass(frozen=True)
class Graph:
    """weighted undirected graph: W symmetric with zero diagonal"""
    W: sp.csr_matrix

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def degree(self) -> np.ndarray:
        return np.asarray(self.W.sum(1)).ravel()

    @property
    def vol(self) -> float:
        return float(self.W.sum())

    def laplacian(self) -> sp.csr_matrix:
        """L = D - W"""
        return (sp.diags(self.degree) - self.W).tocsr()

    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ordered pairs (i, j) with W_ij > 0 and their weights; both directions of each edge appear"""
        coo = self.W.tocoo()
        return coo.row.astype(np.int32), coo.col.astype(np.int32), coo.data

    def n_components(self) -> int:
        return connected_components(self.W, directed=False)[0]


def knn_graph(x: np.ndarray, cfg: GraphConfig) -> Graph:
    """kNN graph with W_ij = exp(-|x_i - x_j|^2 / (sigma_i sigma_j)), sigma_i the distance to i's k-th neighbour"""
    n = len(x)

    # connect each point to its k nearest neighbours and compute weights
    dist, idx = NearestNeighbors(n_neighbors=cfg.k + 1).fit(x).kneighbors(x)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = np.maximum(dist[:, -1], 1e-12)
    w = np.exp(-dist**2 / (sigma[:, None] * sigma[idx]))

    # build sparse adjacency matrix and symmetrize
    A = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), cfg.k), idx.ravel())), shape=(n, n))
    W = A.maximum(A.T)

    # remove self-loops
    W.setdiag(0)
    W.eliminate_zeros()
    graph = Graph(W.tocsr())

    if (c := graph.n_components()) > 1:
        warnings.warn(f"kNN graph has {c} connected components: the eigenvalue 0 has multiplicity {c}")
    return graph
