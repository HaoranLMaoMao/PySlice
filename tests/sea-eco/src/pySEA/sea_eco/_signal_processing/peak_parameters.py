# -*- coding: utf-8 -*-
"""Peak parameter estimators for SEA-eco spectra."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Sequence


def _estimate_FWPM(data: NDArray, E: NDArray, percent: float = 0.5,
                   verbose: str | None = None, return_bnds: bool = False) -> float | tuple[float, float, float]:
    """Estimate full width at ``percent`` of peak maximum.

    Parameters
    ----------
    data : NDArray
        1D data containing a peak.
    E : NDArray
        Energy axis values aligned to ``data``.
    percent : float, optional
        Fractional height at which width is measured, by default 0.5.
    verbose : {'info','all',None}, optional
        Print peak info when set.
    return_bnds : bool, optional
        Return width plus left/right bounds, by default False.

    Returns
    -------
    float | tuple[float, float, float]
        Width at percentage height, optionally with (left, right) bounds.
    """
    
    E_max_i = np.nanargmax(data, axis=-1)
    E_max_v = E[E_max_i]
    if verbose in ['info', 'all']: print(f'E_max: {E_max_v:.2f}')

    I_max = np.nanmax(data, axis=-1)
    I_p = I_max * percent
    if verbose in ['info', 'all']: print(f'I_max: {I_max:.2f}')

    data = np.abs(data-I_p[...,None])

    l = np.nanargmin(data[:E_max_i])
    l = E[:E_max_i][l]
    r = np.nanargmin(data[E_max_i:])
    r = E[E_max_i:][r]
    if return_bnds:
        return r-l, l, r
    else:
        return r-l
    
def _estimate_LSPM(data: NDArray, E: NDArray, percent: float = 0.5,
                   E_max_i: int | None = None,
                   verbose: str | None = None) -> float:
    """Estimate left-side position at ``percent`` of peak maximum."""
    if E_max_i is None:
        E_max_i = int(np.nanargmax(data))
    if verbose in ['info', 'all']:
        print(f'E_max_i: {E_max_i}')

    l_idx = np.nanargmin(np.abs(data[:E_max_i] - percent*np.nanmax(data)))
    return float(E[:E_max_i][l_idx])

def estimate_FWPM(data: NDArray,
                  percent: float = 0.5,
                  E: Sequence[float] | NDArray | None = None,
                  verbose: str | None = None,
                  return_bnds: bool = False) -> NDArray:
    """Estimate full width at percentage maximum for each spectrum.

    Parameters
    ----------
    data : NDArray
        Array containing one or more spectra along the last axis.
    percent : float, optional
        Fractional height at which width is measured, by default 0.5.
    E : Sequence[float] | NDArray | None, optional
        Energy axis values; if None, pixel indices are used, by default None.
    verbose : {'info','all',None}, optional
        Print peak information when set.
    return_bnds : bool, optional
        If True, include left/right bounds, by default False.

    Returns
    -------
    NDArray
        Widths (and optionally bounds) shaped like input spectra.
    """
    if E is None:
        if verbose in ['all']: print('E is None. The returned values will be in relative indicies.')
        E = np.arange(data.shape[-1])

    shape = np.shape(data)
    data = data.reshape(-1, shape[-1])
    if return_bnds: shape = shape[:-1] + tuple([3])

    fwpm = np.asarray([_estimate_FWPM(d, E=E, percent=percent, return_bnds=return_bnds, verbose=verbose) for d in data], dtype=float).reshape(shape)
        
    return fwpm
    
def _estimate_FWPM_center(data: NDArray, E: NDArray, percent: float = 0.5,
                          verbose: str | None = None) -> float:
    """Estimate peak center at ``percent`` of maximum for a single spectrum."""
    
    E_max_i = np.nanargmax(data, axis=-1)
    E_max_v = E[E_max_i]
    if verbose in ['info', 'all']: print(f'E_max: {E_max_v:.2f}')

    I_max = np.nanmax(data, axis=-1)
    I_p = I_max * percent
    if verbose in ['info', 'all']: print(f'I_max: {I_max:.2f}')

    data = np.abs(data-I_p[...,None])

    l = np.nanargmin(data[:E_max_i])
    l = E[:E_max_i][l]
    r = np.nanargmin(data[E_max_i:])
    r = E[E_max_i:][r]

    return float((l+r)/2)

# def estimate_FWPM_center(data):
#     _, l, r = estimate_FWPM(data, percent=0.5, return_sides=True)
#     return (l+r)/2

def estimate_FWPM_center(data: NDArray,
                         percent: float = 0.5,
                         E: Sequence[float] | NDArray | None = None,
                         verbose: str | None = None) -> NDArray:
    """Estimate peak centers at ``percent`` maximum for each spectrum."""
    if E is None:
        if verbose in ['all']: print('E is None. The returned values will be in relative indicies.')
        E = np.arange(data.shape[-1])

    shape = np.shape(data)
    data = data.reshape(-1, shape[-1])
    fwpm = np.asarray([_estimate_FWPM_center(d, E=E, percent=percent, verbose=verbose) for d in data], dtype=float).reshape(shape[:2])
        
    return fwpm

def estimate_skew(data: NDArray,
                  E: Sequence[float] | NDArray | None = None,
                  percent: float = 0.5) -> NDArray:
    """Estimate skewness as center offset relative to peak maximum."""
    return estimate_FWPM_center(data, percent=percent, E=E) - np.nanmax(data, axis=-1)
