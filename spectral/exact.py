import numpy as np
import scipy.linalg

from spectral.graph import Graph


def exact_eigs(graph: Graph, d: int) -> tuple[np.ndarray, np.ndarray]:
    """compute the exact first d eigenvalues and eigenvectors of the graph Laplacian"""
    return scipy.linalg.eigh(graph.laplacian().toarray(), subset_by_index=[0, d - 1])


def compare(U: np.ndarray, V: np.ndarray, lam_hat: np.ndarray, lam: np.ndarray) -> dict:
    """compare learned (U, lam_hat) against exact (V, lam)"""
    cosine = np.abs((U * V).sum(0)) / (np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0))
    subspace = np.linalg.norm(np.linalg.qr(V)[0].T @ np.linalg.qr(U)[0]) ** 2 / U.shape[1]
    return {
        "cosine": cosine,
        "subspace": subspace,
        "eig_abs_err": np.abs(lam_hat - lam),
        "orth_err": np.linalg.norm(U.T @ U / len(U) - np.eye(U.shape[1])),
    }
