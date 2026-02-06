from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib

plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern
plt.rcParams["mathtext.rm"] = "serif"
# get rid of type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# set font size for title, axis labels, and legend, and tick labels
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10


@dataclass(frozen=True)
class DoctrineSpec:
    """Specification for a commitment-distance doctrine distribution."""

    name: str
    alpha: float
    beta: float


def plot_commitment_beta_pdfs(
    R: float,
    R_min: float = 0.0,
    specs: Optional[Sequence[DoctrineSpec]] = None,
    n_points: int = 600,
    ax: Optional[plt.Axes] = None,
    linewidth: float = 2.0,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot research-grade PDFs for scaled Beta commitment distance models.

    The latent commitment distance D is defined by:
        U ~ Beta(alpha, beta) on [0, 1]
        D = R_min + (R - R_min) * U  on [R_min, R]

    This function plots PDFs in terms of the *scaled physical distance*:
        x = D / R  (dimensionless), so the x-axis reads naturally for aerospace audiences.

    Parameters
    ----------
    R : float
        Maximum travel distance (range budget). Must satisfy R > 0.
    R_min : float, default 0.0
        Minimum effective commitment distance. Must satisfy 0 <= R_min < R.
    specs : sequence of DoctrineSpec, optional
        Doctrine distributions to plot. If None, uses four defaults:
            aggressive: (8,2), passive: (2,8), normal: (2,2), uniform: (1,1)
    n_points : int, default 600
        Number of evaluation points for the PDF curves.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If None, a new Figure/Axes is created.
    linewidth : float, default 2.0
        Line width for plotted PDFs.
    show : bool, default True
        Whether to call plt.show().

    Returns
    -------
    fig, ax : (matplotlib.figure.Figure, matplotlib.axes.Axes)

    Notes
    -----
    - The plotted PDFs are for X = D/R supported on [R_min/R, 1].
    - When plotting in x = D/R rather than D, the density differs by a constant scale factor
      (Jacobian) but the *shape comparisons* and doctrine interpretation remain clean.
    """
    # ------------------------
    # Validate inputs
    # ------------------------
    if not np.isfinite(R) or R <= 0:
        raise ValueError(f"R must be finite and > 0. Got R={R}.")
    if not np.isfinite(R_min) or R_min < 0 or R_min >= R:
        raise ValueError(
            f"R_min must satisfy 0 <= R_min < R. Got R_min={R_min}, R={R}."
        )
    if n_points < 50:
        raise ValueError(
            f"n_points should be >= 50 for smooth curves. Got n_points={n_points}."
        )

    # ------------------------
    # Default doctrine set
    # ------------------------
    if specs is None:
        specs = [
            DoctrineSpec("Passive (2,8)", 2.0, 8.0),
            DoctrineSpec("Nominal (2,2)", 2.0, 2.0),
            DoctrineSpec("Aggressive (8,2)", 8.0, 2.0),
        ]

    for s in specs:
        if s.alpha <= 0 or s.beta <= 0:
            raise ValueError(
                f"Alpha/Beta must be > 0 for {s}. Got ({s.alpha},{s.beta})."
            )

    # ------------------------
    # Axes setup
    # ------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 2.5), layout="constrained")
    else:
        fig = ax.figure

    # ------------------------
    # Domain: x = D/R in [R_min/R, 1]
    # Use open interval endpoints to avoid infinite densities for alpha<1 or beta<1 (not here,
    # but safe practice).
    # ------------------------
    x_min = R_min / R
    eps = 1e-6
    x = np.linspace(x_min + eps, 1.0 - eps, n_points)

    # Map to U in [0, 1]: U = (D - R_min) / (R - R_min) = (xR - R_min)/(R - R_min)
    denom = R - R_min
    u = (x * R - R_min) / denom
    u = np.clip(u, 0.0, 1.0)

    # Jacobian for transforming pdf_U(u) to pdf_X(x):
    # D = R x, u = (R x - R_min) / (R - R_min)  => du/dx = R/(R - R_min)
    # pdf_X(x) = pdf_U(u) * |du/dx|
    jac = R / denom

    # ------------------------
    # Plot curves
    # ------------------------
    for s in specs:
        pdf_u = beta.pdf(u, s.alpha, s.beta)
        pdf_x = pdf_u * jac
        ax.plot(x, pdf_x, linewidth=linewidth, label=s.name)

    # ------------------------
    # Labels & styling (no manual colors)
    # ------------------------
    ax.set_xlabel(r"Scaled commitment distance $D/R$")
    ax.set_ylabel(r"PDF of $D/R$")

    ax.set_xlim(0, 1.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    # Light annotation of R_min if nonzero
    if R_min > 0:
        ax.axvline(x_min, linestyle=":", linewidth=1.2)
        ax.text(
            x_min,
            ax.get_ylim()[1] * 0.95,
            r"$R_{\min}/R$",
            rotation=90,
            va="top",
            ha="right",
        )

    # put legend outside the plot under the x-axis
    fig.legend(loc="outside lower center", ncol=3)

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    # Example usage
    plot_commitment_beta_pdfs(R=1.0, R_min=0.25)
