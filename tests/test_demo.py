"""Demo: recover a known time-varying 2-channel spectral matrix.

Ground truth: two independent time-varying MA(1) channels mixed by a
time-varying lower-triangular matrix C(u) = [[1, 0], [c(u), 1]], so
S(u, f) = C(u) diag(S_1, S_2) C(u)^T in closed form (two-sided).
"""

import numpy as np

from multivar_tv_psd import fit

FS = 1.0


def _b1(u):
    return 1.1 * np.cos(1.5 - np.cos(4 * np.pi * u))


def _b2(u):
    return -0.7 + 0.4 * u


def _c(u):
    return 0.8 * np.sin(2 * np.pi * u)


def simulate(n=16384, seed=1):
    rng = np.random.default_rng(seed)
    u = np.arange(n) / n
    eps = rng.standard_normal((n + 1, 2))
    x = np.stack(
        [
            eps[1:, 0] + _b1(u) * eps[:-1, 0],
            eps[1:, 1] + _b2(u) * eps[:-1, 1],
        ],
        axis=1,
    )
    z = x.copy()
    z[:, 1] += _c(u) * x[:, 0]
    return z


def true_spectral_matrix(u, f):
    """(nt, nf, 2, 2) two-sided S(u, f) on the outer grid u x f."""
    uu, ff = np.meshgrid(u, f, indexing="ij")
    dt = 1.0 / FS
    s = np.zeros(uu.shape + (2, 2))
    s1 = dt * (1 + _b1(uu) ** 2 + 2 * _b1(uu) * np.cos(2 * np.pi * ff * dt))
    s2 = dt * (1 + _b2(uu) ** 2 + 2 * _b2(uu) * np.cos(2 * np.pi * ff * dt))
    c = _c(uu)
    s[..., 0, 0] = s1
    s[..., 0, 1] = c * s1
    s[..., 1, 0] = c * s1
    s[..., 1, 1] = c**2 * s1 + s2
    return s


def test_recovers_time_varying_spectral_matrix():
    z = simulate()
    res = fit(
        z,
        fs=FS,
        nseg=64,
        navg_t=8,
        navg_f=2,
        kt=8,
        kf=8,
        num_warmup=300,
        num_samples=300,
        seed=0,
    )
    s_med = res.spectral_median()
    s_true = true_spectral_matrix(res.times / (len(z) / FS), res.freqs)

    # Positive definite at every (t, f) by construction.
    assert np.linalg.eigvalsh(s_med).min() > 0

    # Relative integrated Frobenius error against the closed-form truth.
    rise = np.sum(np.abs(s_med - s_true) ** 2) / np.sum(np.abs(s_true) ** 2)
    assert rise < 0.25, f"RISE={rise:.3f}"

    # The mixing is real, so Im S_21 = 0: the free Im(theta_21) surface must
    # shrink to zero, not hallucinate structure.
    im_scale = np.max(np.abs(s_med[..., 1, 0].imag)) / np.max(np.abs(s_true[..., 1, 0]))
    assert im_scale < 0.1, f"spurious Im S_21 at {im_scale:.3f} of cross-spectrum scale"

    # The recovered cross-spectrum must track the sign flip of c(u) over time
    # (a stationary model cannot). Since S_21 = c(u) S_11, the ratio
    # Re S_21 / S_11 estimates c(u) directly; average over frequency.
    c_true = _c(res.times / (len(z) / FS))
    c_fit = (s_med[..., 1, 0].real / s_med[..., 0, 0].real).mean(axis=1)
    corr = np.corrcoef(c_true, c_fit)[0, 1]
    assert corr > 0.9, f"cross-spectrum time-tracking corr={corr:.3f}"
