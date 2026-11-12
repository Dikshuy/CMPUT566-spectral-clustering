import matplotlib
import numpy as np
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectral.config import Config
from spectral.train import run


def plot_eigvecs(res: dict, path: str) -> None:
    """exact (top row) vs learned (bottom row) eigenvectors over the data"""
    x, V, U = res["x"], res["V"], res["U"][0]
    V, U = V / np.linalg.norm(V, axis=0), U / np.linalg.norm(U, axis=0)
    U = U * np.sign((U * V).sum(0))
    d = V.shape[1]
    fig, axes = plt.subplots(2, d, figsize=(2.2 * d, 4.6), squeeze=False)
    for i in range(d):
        lim = np.abs(V[:, i]).max()
        for row, (vecs, label) in enumerate([(V, "exact"), (U, "ALLO")]):
            ax = axes[row, i]
            ax.scatter(x[:, 0], x[:, 1], c=vecs[:, i], s=2, cmap="coolwarm", vmin=-lim, vmax=lim)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label} $u_{{{i + 1}}}$", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    res = run(tyro.cli(Config))
    plot_eigvecs(res, "eigvecs.png")
    print("[saved] eigvecs.png")
