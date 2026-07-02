"""STFT tiling and 2D local averaging of the multichannel periodogram."""

import numpy as np


def stft_tiles(
    z: np.ndarray, fs: float, nseg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Non-overlapping rectangular-window STFT of a p-channel series.

    Uses the Whittle convention d = dt * sum_t z_t exp(-2pi i k t / nseg),
    so that d(t_n, f_m) ~ CN(0, T_b * S(t_n, f_m)) with S two-sided.

    Args:
        z: (n, p) real time series.
        fs: Sampling frequency.
        nseg: Samples per segment; trailing remainder is dropped.

    Returns:
        (d, times, freqs, Tb): d is (nt, nf, p) complex with DC and Nyquist
        dropped, times are segment centres [s], freqs the Fourier
        frequencies [Hz], Tb the segment duration.
    """
    n, _ = z.shape
    dt = 1.0 / fs
    nt = n // nseg
    seg = z[: nt * nseg].reshape(nt, nseg, -1)
    d = dt * np.fft.fft(seg, axis=1)
    k = np.arange(1, nseg // 2)
    times = (np.arange(nt) + 0.5) * nseg * dt
    return d[:, k, :], times, k * fs / nseg, nseg * dt


def coarse_grain(
    d: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    navg_t: int = 1,
    navg_f: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sum rank-1 tile Wisharts over navg_t x navg_f neighbourhoods.

    Each super-tile Y = sum d d* ~ CW_p(Tb S, N_loc) is eigendecomposed into
    rescaled eigenvectors u_nu = sqrt(lam_nu) v_nu, the form consumed by the
    per-channel regression likelihood. navg_t = navg_f = 1 recovers the
    rank-1 case (u_1 = d up to phase).

    Returns:
        (u, t_centres, f_centres, dof): u is (nt_c, nf_c, p, p) complex with
        eigenvector nu in u[..., :, nu]; dof = navg_t * navg_f.
    """
    nt, nf, p = d.shape
    nt_c, nf_c = nt // navg_t, nf // navg_f
    dd = d[: nt_c * navg_t, : nf_c * navg_f].reshape(nt_c, navg_t, nf_c, navg_f, p)
    y = np.einsum("tafbj,tafbl->tfjl", dd, dd.conj())
    lam, v = np.linalg.eigh(y)
    u = v * np.sqrt(np.clip(lam, 0.0, None))[..., None, :]
    t_c = times[: nt_c * navg_t].reshape(nt_c, navg_t).mean(axis=1)
    f_c = freqs[: nf_c * navg_f].reshape(nf_c, navg_f).mean(axis=1)
    return u, t_c, f_c, navg_t * navg_f
