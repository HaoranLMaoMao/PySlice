"""Normalization helpers for SEA-eco signals."""

#Imports: Typing
from __future__ import annotations
from typing import Literal, Tuple, Union
from numpy.typing import NDArray

#Imports: External
import numpy as np
from tqdm import tqdm

#Imports: Internal


def nv_correction(data: NDArray,
                  type: Literal['abs', 'offset'] = 'abs',
                  dims: Union[None, int, Tuple[int, ...]] = None) -> NDArray:
    """Correct negative values by absolute value or offsetting.

    Parameters
    ----------
    data : NDArray
        Data to be corrected.
    type : {'abs', 'offset'}, optional
        ``'abs'`` takes absolute value; ``'offset'`` subtracts the minimum
        value, by default 'abs'.
    dims : None | int | tuple[int, ...], optional
        Axes along which to compute the minimum when offsetting. ``None`` uses
        the full array. Defaults to None.

    Returns
    -------
    NDArray
        Corrected data.
    """
    if type == 'abs':
         return np.abs(data)
    if type == 'offset':
        min_val = np.nanmin(data if dims is None else np.nanmin(data, axis=dims, keepdims=True))
        return data - min_val
    raise AttributeError(f"{type!r} is not a valid correction type.")
    
def normalize_by_ZLP(data: NDArray, threshold: float = 3.,
                     type: Literal['area', 'max'] = 'area',
                     zero_base: bool = True, dims: int = -1) -> NDArray:
    """Normalize spectra relative to the zero-loss peak region."""
    d = data[:threshold]

    ZLP_I = normalize(d, type=type, zero_base=zero_base, dims=dims)

    return data/ZLP_I

def normalize(data: NDArray,
              type: Literal['area', 'max'] = 'area',
              zero_base: bool = True,
              dims: int = -1) -> NDArray:
    """Normalize data by area or maximum along a dimension.

    Parameters
    ----------
    data : NDArray
        Data to be normalized.
    type : {'area', 'max'}, optional
        ``'area'`` scales integral to 1; ``'max'`` scales maximum to 1, by
        default 'area'.
    zero_base : bool, optional
        Subtract minimum before normalization, by default True.
    dims : int, optional
        Axis along which to normalize, by default -1.

    Returns
    -------
    NDArray
        Normalized data.
    """
    
    if zero_base:
        min_val = np.nanmin(data)
        data = data - min_val
    if type == 'area': 
        return data/np.nansum(data, axis=dims)
    if type == 'max':
         return data/np.nanmax(data, axis=dims)
    raise AttributeError(f"{type!r} is not a valid normalization type.")
