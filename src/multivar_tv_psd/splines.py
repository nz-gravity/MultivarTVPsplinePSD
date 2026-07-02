"""Tensor-product P-spline building blocks: bases, penalties, eigen-whitening."""

import numpy as np
from scipy.interpolate import BSpline


def bspline_basis(x: np.ndarray, num_basis: int, degree: int = 3) -> np.ndarray:
    """Clamped B-spline design matrix on [0, 1] with uniform interior knots.

    Args:
        x: Evaluation points in [0, 1].
        num_basis: Number of basis functions K (K >= degree + 1).
        degree: Spline degree.

    Returns:
        Dense (len(x), K) design matrix.
    """
    n_interior = num_basis - degree - 1
    if n_interior < 0:
        raise ValueError("num_basis must be at least degree + 1")
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    x = np.clip(x, 0.0, 1.0)
    return BSpline.design_matrix(x, knots, degree, extrapolate=False).toarray()


def difference_penalty(num_basis: int, order: int = 2) -> np.ndarray:
    """P = D_order^T D_order, the standard P-spline difference penalty."""
    d = np.diff(np.eye(num_basis), n=order, axis=0)
    return d.T @ d


def whitened_basis(
    grid: np.ndarray, num_basis: int, degree: int = 3, penalty_order: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Basis rotated into the penalty eigenbasis.

    Returns:
        (B @ U, lam) where P = U diag(lam) U^T. A surface with coefficients
        Z in this basis is Lambda = (B_t U_t) Z (B_f U_f)^T and the
        Kronecker-sum penalty is diagonal: d_ab = phi_t lam_t[a] + phi_f lam_f[b].
    """
    basis = bspline_basis(grid, num_basis, degree)
    lam, eigvec = np.linalg.eigh(difference_penalty(num_basis, penalty_order))
    return basis @ eigvec, np.clip(lam, 0.0, None)
