import numpy as np
import scipy.linalg
import scipy.optimize

from spectral.graph import Graph


def exact_eigs(graph: Graph, d: int) -> tuple[np.ndarray, np.ndarray]:
    """compute the exact first d eigenvalues and eigenvectors of the graph Laplacian"""
    return scipy.linalg.eigh(graph.laplacian().toarray(), subset_by_index=[0, d - 1])


def compare(U: np.ndarray, V: np.ndarray, lam_hat: np.ndarray, lam: np.ndarray, L) -> dict:
    """compare learned (U, lam_hat) against exact (V, lam)"""
    C = np.abs((U / np.linalg.norm(U, axis=0)).T @ (V / np.linalg.norm(V, axis=0)))     # C[i, j] = |cos(u_i, v_j)|
    _, match = scipy.optimize.linear_sum_assignment(C, maximize=True)
    subspace = np.linalg.norm(np.linalg.qr(V)[0].T @ np.linalg.qr(U)[0]) ** 2 / U.shape[1]
    rayleigh = (U * (L @ U)).sum(0) / (U * U).sum(0)
    return {
        "cosine": np.diag(C),
        "cosine_matched": C[np.arange(len(C)), match],
        "match": match,
        "subspace": subspace,
        "eig_abs_err": np.abs(lam_hat - lam),
        "rayleigh": rayleigh,
        "rayleigh_abs_err": np.abs(rayleigh - lam),
        "orth_err": np.linalg.norm(U.T @ U / len(U) - np.eye(U.shape[1])),
    }
