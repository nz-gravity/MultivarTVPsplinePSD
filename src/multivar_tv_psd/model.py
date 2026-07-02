"""NumPyro model, fitting, and posterior spectral-matrix reconstruction.

Parametrises S(t,f)^{-1} = T*(t,f) D^{-1}(t,f) T(t,f); each Cholesky
component (log delta^2_j, Re theta_jl, Im theta_jl) is a tensor-product
P-spline surface with its own anisotropic Kronecker-sum penalty, sampled in
the whitened penalty eigenbasis.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

from .periodogram import coarse_grain, stft_tiles
from .splines import whitened_basis

# Surface ordering: p log-diagonals, then p(p-1)/2 Re(theta), then Im(theta),
# with off-diagonals in np.tril_indices(p, -1) order.


def tv_psd_model(
    u: jnp.ndarray,
    dof: int,
    Tb: float,
    BUt: jnp.ndarray,
    BUf: jnp.ndarray,
    lam_t: jnp.ndarray,
    lam_f: jnp.ndarray,
    p: int,
    eta: float = 1.0,
    tau0: float = 1e-4,
    alpha_phi: float = 2.0,
    beta_phi: float = 1.0,
):
    """Whitened tensor-product P-spline model for the time-varying CSD matrix.

    Args:
        u: (nt, nf, p, p) complex rescaled eigenvectors of the coarse-grained
            Wishart statistic (eigenvector nu in u[..., :, nu]).
        dof: Wishart degrees of freedom per super-tile.
        Tb: STFT segment duration.
        BUt, BUf: whitened basis matrices (nt, Kt), (nf, Kf).
        lam_t, lam_f: penalty eigenvalues.
        p: Number of channels.
        eta: Safe-Bayes likelihood tempering in (0, 1].
        tau0: Fixed weak precision for the penalty null space.
        alpha_phi, beta_phi: Gamma hyperprior on the smoothing precisions.
    """
    ncomp = p * p
    kt, kf = BUt.shape[1], BUf.shape[1]

    phi = numpyro.sample(
        "phi", dist.Gamma(alpha_phi, beta_phi).expand([ncomp, 2]).to_event(2)
    )
    s = numpyro.sample("s", dist.Normal().expand([ncomp, kt, kf]).to_event(3))

    prec = (
        phi[:, 0, None, None] * lam_t[None, :, None]
        + phi[:, 1, None, None] * lam_f[None, None, :]
        + tau0
    )
    surf = jnp.einsum("tk,ckl,fl->ctf", BUt, s / jnp.sqrt(prec), BUf)
    numpyro.deterministic("surfaces", surf)

    log_delta2 = jnp.moveaxis(surf[:p], 0, -1)  # (nt, nf, p)
    q = p * (p - 1) // 2
    rows, cols = np.tril_indices(p, -1)
    theta = jnp.zeros(surf.shape[1:] + (p, p), dtype=complex)
    theta = theta.at[..., rows, cols].set(
        jnp.moveaxis(surf[p : p + q] + 1j * surf[p + q :], 0, -1)
    )

    # Per-channel regression: r_j = u_j - sum_{l<j} theta_jl u_l, summed over
    # eigenvectors nu (Eq. 20 of the stationary paper, per tile).
    resid = jnp.sum(jnp.abs(u - theta @ u) ** 2, axis=-1)  # (nt, nf, p)
    loglik = jnp.sum(-dof * log_delta2 - resid / (Tb * jnp.exp(log_delta2)))
    numpyro.factor("whittle", eta * loglik)


def spectral_matrix(surfaces: np.ndarray, p: int) -> np.ndarray:
    """Reconstruct S(t,f) from Cholesky-component surfaces.

    Args:
        surfaces: (..., p^2, nt, nf) real surfaces in model ordering.
        p: Number of channels.

    Returns:
        (..., nt, nf, p, p) complex Hermitian positive-definite matrices
        (two-sided convention).
    """
    q = p * (p - 1) // 2
    delta2 = np.exp(np.moveaxis(surfaces[..., :p, :, :], -3, -1))
    theta = np.moveaxis(
        surfaces[..., p : p + q, :, :] + 1j * surfaces[..., p + q :, :, :], -3, -1
    )
    t_mat = np.zeros(delta2.shape[:-1] + (p, p), dtype=complex)
    t_mat[..., np.arange(p), np.arange(p)] = 1.0
    rows, cols = np.tril_indices(p, -1)
    t_mat[..., rows, cols] = -theta
    t_inv = np.linalg.inv(t_mat)
    return (t_inv * delta2[..., None, :]) @ t_inv.conj().swapaxes(-1, -2)


@dataclass
class TVPSDResult:
    """Posterior over the time-varying spectral density matrix."""

    mcmc: MCMC
    times: np.ndarray
    freqs: np.ndarray
    p: int

    def spectral_samples(self) -> np.ndarray:
        """(nsamples, nt, nf, p, p) posterior draws of S(t,f), two-sided."""
        surf = np.asarray(self.mcmc.get_samples()["surfaces"])
        return spectral_matrix(surf, self.p)

    def spectral_median(self) -> np.ndarray:
        """Elementwise posterior median of S(t,f) (Hermitian, PD in practice)."""
        s = self.spectral_samples()
        med = np.median(s.real, axis=0) + 1j * np.median(s.imag, axis=0)
        return (med + med.conj().swapaxes(-1, -2)) / 2


def fit(
    z: np.ndarray,
    fs: float,
    nseg: int,
    navg_t: int = 1,
    navg_f: int = 1,
    kt: int = 10,
    kf: int = 10,
    degree: int = 3,
    eta: float = 1.0,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    seed: int = 0,
    **model_kwargs,
) -> TVPSDResult:
    """Fit the time-varying multivariate PSD model to a p-channel series.

    Args:
        z: (n, p) real time series.
        fs: Sampling frequency.
        nseg: STFT segment length.
        navg_t, navg_f: Local-averaging window (time x frequency tiles).
        kt, kf: Basis dimensions in time and frequency.
        degree: B-spline degree.
        eta: Safe-Bayes tempering.
        num_warmup, num_samples, num_chains, seed: NUTS settings.
        **model_kwargs: Passed through to tv_psd_model (tau0, alpha_phi, ...).

    Returns:
        TVPSDResult with the fitted MCMC and the coarse (t, f) grid.
    """
    z = np.asarray(z)
    p = z.shape[1]
    d, times, freqs, Tb = stft_tiles(z, fs, nseg)
    u, t_c, f_c, dof = coarse_grain(d, times, freqs, navg_t, navg_f)

    t01 = (t_c - t_c[0]) / (t_c[-1] - t_c[0])
    f01 = (f_c - f_c[0]) / (f_c[-1] - f_c[0])
    BUt, lam_t = whitened_basis(t01, kt, degree)
    BUf, lam_f = whitened_basis(f01, kf, degree)

    mcmc = MCMC(
        NUTS(tv_psd_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(
        random.PRNGKey(seed),
        u=jnp.asarray(u),
        dof=dof,
        Tb=Tb,
        BUt=jnp.asarray(BUt),
        BUf=jnp.asarray(BUf),
        lam_t=jnp.asarray(lam_t),
        lam_f=jnp.asarray(lam_f),
        p=p,
        eta=eta,
        **model_kwargs,
    )
    return TVPSDResult(mcmc=mcmc, times=t_c, freqs=f_c, p=p)
