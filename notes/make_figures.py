"""Generate the ms.tex figures from the demo benchmark (tests/test_demo.py).

Run from the repo root: uv run python notes/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_demo import FS, _c, simulate, true_spectral_matrix  # noqa: E402

from multivar_tv_psd import fit  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

z = simulate()
res = fit(
    z, fs=FS, nseg=64, navg_t=8, navg_f=2, kt=8, kf=8,
    num_warmup=300, num_samples=300, seed=0,
)
u_grid = res.times / (len(z) / FS)
s_samp = res.spectral_samples()
s_med = res.spectral_median()
s_true = true_spectral_matrix(u_grid, res.freqs)

rise = np.sum(np.abs(s_med - s_true) ** 2) / np.sum(np.abs(s_true) ** 2)
print(f"RISE = {rise:.4f}")

# Figure 1: truth vs posterior-median surfaces.
panels = [
    (r"$\log S_{11}$", lambda s: np.log(s[..., 0, 0].real)),
    (r"$\Re\,S_{21}$", lambda s: s[..., 1, 0].real),
    (r"$\Im\,S_{21}$", lambda s: np.imag(s[..., 1, 0])),
    (r"$\log S_{22}$", lambda s: np.log(s[..., 1, 1].real)),
]
fig, axes = plt.subplots(4, 2, figsize=(6.5, 9), sharex=True, sharey=True, layout="constrained")
for row, (label, comp) in enumerate(panels):
    truth, est = comp(s_true), comp(s_med)
    vmin = min(truth.min(), est.min())
    vmax = max(truth.max(), est.max())
    for col, data in enumerate([truth, est]):
        im = axes[row, col].pcolormesh(
            res.freqs, u_grid, data, vmin=vmin, vmax=vmax, cmap="viridis", rasterized=True
        )
    fig.colorbar(im, ax=axes[row, :], label=label, shrink=0.9)
axes[0, 0].set_title("Truth")
axes[0, 1].set_title("Posterior median")
for ax in axes[-1]:
    ax.set_xlabel("Frequency [Hz]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"Rescaled time $u$")
fig.savefig(OUT / "surfaces.pdf")

# Figure 2: time-tracking of the mixing c(u) via Re S_21 / S_11.
ratio = (s_samp[..., 1, 0].real / s_samp[..., 0, 0].real).mean(axis=2)  # (ns, nt)
lo, med, hi = np.quantile(ratio, [0.05, 0.5, 0.95], axis=0)
fig, ax = plt.subplots(figsize=(5, 3.2), layout="constrained")
ax.plot(u_grid, _c(u_grid), "k--", label=r"true $c(u)$")
ax.plot(u_grid, med, "C0", label="posterior median")
ax.fill_between(u_grid, lo, hi, color="C0", alpha=0.3, label="90% CI")
ax.axhline(0.0, color="C3", lw=1, label="stationary fit (time-averaged)")
ax.set_xlabel(r"Rescaled time $u$")
ax.set_ylabel(r"$\Re\,S_{21}/S_{11}$")
ax.legend(fontsize=8)
fig.savefig(OUT / "cross_tracking.pdf")
print(f"Figures written to {OUT}")
