"""Time-varying multivariate spectral density estimation with tensor-product P-splines."""

from .model import TVPSDResult, fit, spectral_matrix, tv_psd_model
from .periodogram import coarse_grain, stft_tiles
from .splines import bspline_basis, difference_penalty, whitened_basis

__all__ = [
    "TVPSDResult",
    "fit",
    "spectral_matrix",
    "tv_psd_model",
    "coarse_grain",
    "stft_tiles",
    "bspline_basis",
    "difference_penalty",
    "whitened_basis",
]
