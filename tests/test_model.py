"""Fast consistency checks of the likelihood and parametrisation math."""

import numpy as np

from multivar_tv_psd import coarse_grain, spectral_matrix, stft_tiles


def test_stft_whittle_scaling():
    # White noise sigma^2: two-sided S = sigma^2 * dt, so E|d|^2 = Tb * S.
    rng = np.random.default_rng(0)
    sigma, fs = 2.0, 4.0
    z = sigma * rng.standard_normal((2**16, 1))
    d, _, _, tb = stft_tiles(z, fs=fs, nseg=64)
    s_hat = np.mean(np.abs(d) ** 2) / tb
    assert abs(s_hat / (sigma**2 / fs) - 1) < 0.05


def test_coarse_grain_reconstructs_wishart():
    rng = np.random.default_rng(1)
    d = rng.standard_normal((8, 6, 3)) + 1j * rng.standard_normal((8, 6, 3))
    u, _, _, dof = coarse_grain(d, np.arange(8.0), np.arange(6.0), navg_t=4, navg_f=2)
    assert dof == 8
    y = np.einsum("tafbj,tafbl->tfjl", d.reshape(2, 4, 3, 2, 3), d.reshape(2, 4, 3, 2, 3).conj())
    np.testing.assert_allclose(u @ u.conj().swapaxes(-1, -2), y, atol=1e-10)


def _random_cholesky_surfaces(rng, p, nt, nf):
    return rng.standard_normal((p * p, nt, nf)) * 0.3


def test_spectral_matrix_inverts_parametrisation():
    # S from spectral_matrix must satisfy S^{-1} = T* D^{-1} T.
    rng = np.random.default_rng(2)
    p, nt, nf = 3, 4, 5
    surf = _random_cholesky_surfaces(rng, p, nt, nf)
    s = spectral_matrix(surf, p)

    q = p * (p - 1) // 2
    delta2 = np.exp(np.moveaxis(surf[:p], 0, -1))
    theta = np.moveaxis(surf[p : p + q] + 1j * surf[p + q :], 0, -1)
    t_mat = np.tile(np.eye(p, dtype=complex), (nt, nf, 1, 1))
    t_mat[..., *np.tril_indices(p, -1)] = -theta
    s_inv = t_mat.conj().swapaxes(-1, -2) @ (t_mat / delta2[..., :, None])
    np.testing.assert_allclose(s @ s_inv, np.tile(np.eye(p), (nt, nf, 1, 1)), atol=1e-10)


def test_factorised_likelihood_matches_wishart_trace():
    # The per-channel regression sum equals -N log|S| - tr(S^{-1} Y)/Tb.
    rng = np.random.default_rng(3)
    p, nt, nf, dof, tb = 3, 4, 5, 2, 7.0
    d = rng.standard_normal((nt * 2, nf, p)) + 1j * rng.standard_normal((nt * 2, nf, p))
    u, _, _, _ = coarse_grain(d, np.arange(float(nt * 2)), np.arange(float(nf)), navg_t=2)

    surf = _random_cholesky_surfaces(rng, p, nt, nf)
    s = spectral_matrix(surf, p)
    y = u @ u.conj().swapaxes(-1, -2)
    direct = np.sum(
        -dof * np.log(np.linalg.det(s).real)
        - np.trace(np.linalg.solve(s, y), axis1=-2, axis2=-1).real / tb
    )

    # Reproduce the model's factorised likelihood.
    q = p * (p - 1) // 2
    log_delta2 = np.moveaxis(surf[:p], 0, -1)
    theta = np.zeros((nt, nf, p, p), dtype=complex)
    theta[..., *np.tril_indices(p, -1)] = np.moveaxis(surf[p : p + q] + 1j * surf[p + q :], 0, -1)
    resid = np.sum(np.abs(u - theta @ u) ** 2, axis=-1)
    factorised = np.sum(-dof * log_delta2 - resid / (tb * np.exp(log_delta2)))

    np.testing.assert_allclose(factorised, direct, rtol=1e-8)
