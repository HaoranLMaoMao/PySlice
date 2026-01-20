
"""Signal processing helpers for SEA-eco.

This module provides common utilities for array reshaping, interpolation-based
shifting, and correlation routines used throughout SEA-eco.
"""

#Imports: Typing
from __future__ import annotations
from typing import Callable, Literal, Sequence, Tuple
from numpy.typing import NDArray

#Imports: External
import numpy as np
from tqdm import tqdm
from scipy.interpolate import (CubicSpline, LinearNDInterpolator,
                               RegularGridInterpolator, make_interp_spline)
from scipy.signal import correlate, correlation_lags

#Imports: Internal

def unfold_dims(data: NDArray | Sequence, dims: int | Sequence[int],
                reverse: bool = False,
                return_shapes: bool = False) -> NDArray | tuple[NDArray, list[tuple[int, ...]]]:
    """Collapse or unfold dimensions to ease per-axis operations.

    Parameters
    ----------
    data : NDArray | Sequence
        Input array-like data.
    dims : int | Sequence[int]
        Dimension(s) to move to the end before reshaping; negative values are
        resolved against ``data.ndim``.
    reverse : bool, optional
        If ``True``, collapse non-target axes; otherwise collapse the target
        axes. By default False.
    return_shapes : bool, optional
        Return the intermediate shapes used for folding/unfolding, by default
        False.

    Returns
    -------
    NDArray | tuple[NDArray, list[tuple[int, ...]]]
        Reshaped array, optionally alongside the original and collapsed shapes.

    Raises
    ------
    ValueError
        If ``dims`` contains indices outside the bounds of ``data.ndim``.
    """
    data = np.asarray(data)

    ndim = data.ndim
    if isinstance(dims, int): dims = (dims if dims >= 0 else ndim + dims,)
    else: dims = tuple(d if d >= 0 else ndim + d for d in dims)
    if any(ax < 0 or ax >= ndim for ax in dims):
        raise ValueError("dims contains an out-of-bounds axis.")

    data = np.moveaxis(data, dims, range(-len(dims),0)) # Move the dims to back
    shapes = [np.shape(data)[:-len(dims)],np.shape(data)[-len(dims):]]
    if reverse: 
        new_shape = (-1, *shapes[1]) # Collapse non-dim axes
    else: 
        new_shape = (*shapes[0], -1) # Collapse dim axes
    data = data.reshape(new_shape)

    if return_shapes: return data, shapes
    else: return data

def maxND(data: NDArray, dims: tuple = (-2,-1)) -> NDArray:
    """Find the maximum value and its index across selected dimensions.

    Parameters
    ----------
    data : NDArray
        Input data.
    dims : tuple, optional
        Axes to search for maxima, by default (-2, -1).

    Returns
    -------
    NDArray
        Array shaped like ``(*data.shape_without_dims, len(dims) + 1)`` where
        the final entries hold the indices for each searched dimension followed
        by the maximum value.
    """

    data, shapes = unfold_dims(data, dims, return_shapes=True)

    idx = np.argmax(data, axis=-1) #TODO allow subepixel
    val = np.take(data, idx)
    idx = np.unravel_index(idx, shapes[1])
    return np.stack(idx+(val,), axis=-1, dtype=float) #? should this be a structured array with dtype=((int,)*ndim+(float,))

def shift(data: NDArray, shifts: NDArray,
          dims: int | Sequence[int] = -1,
          interpolate_method: Callable | Literal['linear', 'cubic'] = 'cubic',
          kwargs_interpolate: dict | None = None,
          ) -> NDArray:
    """Shift an array along one or more dimensions using interpolation.

    Parameters
    ----------
    data : NDArray
        Data to shift.
    shifts : NDArray | Sequence[NDArray | float | int]
        Shift amounts. For multiple ``dims`` supply a sequence aligned to
        ``dims``; each element can be scalar or array broadcastable to the data
        shape (or broadcastable after expanding singleton dimensions).
    dims : int | Sequence[int], optional
        Dimension(s) (axis/axes) to shift along, by default -1.
    interpolate_method : Callable | Literal['linear', 'cubic'], optional
        Interpolation selector. Strings map to standard options; callables are
        instantiated directly. Defaults to 'cubic'.
    kwargs_interpolate : dict | None, optional
        Extra keyword arguments passed to the interpolation routine.

    Returns
    -------
    NDArray
        Shifted data array.
    """
    kwargs_interp = {} if kwargs_interpolate is None else dict(kwargs_interpolate)
    data = np.asarray(data)
    ndim = data.ndim

    if isinstance(dims, int):
        dims_tuple = (dims if dims >= 0 else ndim + dims,)
    else:
        dims_tuple = tuple(d if d >= 0 else ndim + d for d in dims)

    if any(ax < 0 or ax >= ndim for ax in dims_tuple):
        raise ValueError("dims contains an out-of-bounds axis.")

    if len(dims_tuple) > 1:
        if isinstance(shifts, Sequence) and len(shifts) == len(dims_tuple):
            shifts_seq = [np.asarray(s) for s in shifts]
        else:
            shifts_arr = np.asarray(shifts)
            if shifts_arr.shape and shifts_arr.shape[-1] == len(dims_tuple):
                shifts_seq = [np.take(shifts_arr, i, axis=-1) for i in range(len(dims_tuple))]
            else:
                raise ValueError("For multiple dims, provide a sequence of shifts matching dims or an array whose last dimension matches len(dims).")
    else:
        shifts_seq = [np.asarray(shifts)]

    def _axis_values(i: int) -> NDArray:
        """Get coordinate values for axis i (assumes ordered grid)."""
        return np.arange(data.shape[i])

    def _broadcast_shift_to_shape(shift_arr: NDArray, axis: int, shape: tuple[int, ...]) -> NDArray:
        """Broadcast a per-axis shift array to the full data shape."""
        arr = np.asarray(shift_arr)
        if arr.shape == ():  # scalar
            return np.full(shape, arr)
        if arr.ndim == 1 and arr.shape[0] == shape[axis]:
            reshape_shape = [1] * len(shape)
            reshape_shape[axis] = shape[axis]
            arr = arr.reshape(reshape_shape)
        elif arr.shape == tuple(shape[:arr.ndim]):
            # allow navigation-shaped shift (e.g., shape without the shifted axis lengths)
            arr = arr.reshape((*arr.shape, *([1] * (len(shape) - arr.ndim))))
        try:
            return np.broadcast_to(arr, shape)
        except ValueError as exc:
            raise ValueError(f"Shift for axis {axis} is not broadcastable to data shape {shape}.") from exc

    # 1D case (single dimension data)
    if data.ndim == 1 or (len(dims_tuple) == 1 and all(s == 1 for i, s in enumerate(data.shape) if i != dims_tuple[0])):
        axis = dims_tuple[0]
        axis_coords = _axis_values(axis)
        shift_b = _broadcast_shift_to_shape(shifts_seq[0], axis, data.shape)
        target_axis = axis_coords - shift_b
        method_name = interpolate_method.lower() if isinstance(interpolate_method, str) else None

        if callable(interpolate_method) and not isinstance(interpolate_method, str):
            interpolator = interpolate_method(axis_coords, data, **kwargs_interp)
            shifted = interpolator(target_axis)
        elif method_name == 'cubic':
            interpolator = CubicSpline(axis_coords, data, **kwargs_interp)
            shifted = interpolator(target_axis)
        elif method_name == 'linear' and data.ndim == 1:
            shifted = np.interp(target_axis, axis_coords, data, **kwargs_interp)
        else:
            spline_kwargs = dict(kwargs_interp)
            spline_kwargs.setdefault('k', 3 if method_name == 'cubic' else 1)
            interpolator = make_interp_spline(axis_coords, data, **spline_kwargs)
            shifted = interpolator(target_axis)
        return shifted

    # ND case
    axes_coords = [_axis_values(i) for i in range(data.ndim)]
    mesh = np.meshgrid(*axes_coords, indexing='ij')
    target_mesh = list(mesh)
    for ax, shift_arr in zip(dims_tuple, shifts_seq):
        shift_b = _broadcast_shift_to_shape(shift_arr, ax, data.shape)
        target_mesh[ax] = target_mesh[ax] - shift_b

    unstructured = any((np.asarray(ax).ndim != 1 or np.asarray(ax).shape[0] != data.shape[i])
                       for i, ax in enumerate(axes_coords))
    source_points = np.stack([m.ravel() for m in mesh], axis=-1)
    target_points = np.stack([g.ravel() for g in target_mesh], axis=-1)

    if not unstructured:
        interp_kwargs = dict(kwargs_interp)
        interp_kwargs.setdefault('bounds_error', False)
        interp_kwargs.setdefault('fill_value', np.nan)
        if isinstance(interpolate_method, str):
            method_name = interpolate_method.lower()
        else:
            method_name = None
        if method_name in {'linear', 'nearest'}:
            interp_kwargs.setdefault('method', method_name)
        structured_interpolator = RegularGridInterpolator(axes_coords, data, **interp_kwargs)
        shifted_flat = structured_interpolator(target_points)
    else:
        #Project: Needs tests for unstructured dimension handling.
        interp_kwargs = dict(kwargs_interp)
        interpolator = LinearNDInterpolator(source_points, data.ravel(), **interp_kwargs)
        shifted_flat = interpolator(target_points)

    return shifted_flat.reshape(data.shape)

def correlate_1D_in_ND(data: NDArray, ref_data : NDArray, 
                       dims: int = -1,
                       normalize: bool = True, kwargs_correlate: dict | None = None
                       ) -> tuple[NDArray, NDArray]:
    """Perform 1D cross-correlation across an N-D signal.

    ``ref_data`` is treated as the reference; positive shifts in ``data`` with
    respect to ``ref_data`` yield positive lags.

    Parameters
    ----------
    data : Signal | NDArray
        N-dimensional array to correlate.
    ref_data : Signal | NDArray
        Reference dataset to correlate against.
    dims : int, optional
        Dimension to correlate along, by default -1.
    normalize : bool, optional
        Perform normalized cross-correlation, by default True.
    kwargs_correlate : dict, optional
        Extra keyword arguments passed to ``scipy.signal.correlate``. By
        default None.

    Returns
    -------
    Lags: NDArray
        Lags.
    Correlations: NDArray
        Correlation coefficients.

    See Also
    --------
    scipy.signal.correlate
    """

    kwargs_correlate = {} if kwargs_correlate is None else dict(kwargs_correlate)

    dim_lens = [np.shape(s)[dims] for s in (data, ref_data)]
    lags = correlation_lags(*dim_lens)

    # Move correlation axis to end
    data, shapes = unfold_dims(data, dims, reverse=True, return_shapes=True)
    ref_data = unfold_dims(ref_data, dims, reverse=True)

    corr = []
    for s1, s2 in tqdm(zip(data, ref_data)):
        if normalize:
            s1 = (s1 - np.nanmean(s1)) / np.nanstd(s1)
            s2 = (s2 - np.nanmean(s2)) / np.nanstd(s2)
        corr.append(correlate(s1, s2, **kwargs_correlate) / min(np.size(s1), np.size(s2)))
    corr = np.stack(corr)
    corr_shape = tuple(shapes[0]) + (corr.shape[-1],)
    corr = corr.reshape(corr_shape)
    
    return lags, corr

def autocorrelate(data: NDArray, dims: Sequence[int] | int = -1,
                  normalize: bool = True,
                  kwargs_correlate: dict | None = None
                  ) -> tuple[list[NDArray], NDArray]:
    """Measure autocorrelation of a signal across specified dimensions.

    Parameters
    ----------
    data : NDArray
        Input signal array.
    dims : Sequence[int] | int, optional
        Dimension(s) to correlate along, by default -1
    normalize : bool, optional
        Perform normalized cross-correlation, by default True
    kwargs_correlate : dict, optional
        Keyword arguments passed to ``scipy.signal.correlate``, by default
        None.

    Returns
    -------
    lags : list[NDArray]
        Lag arrays for each dimension.
    corr : NDArray
        Autocorrelation coefficients.
    """ 
    
    kwargs_correlate = {} if kwargs_correlate is None else dict(kwargs_correlate)

    if isinstance(dims, int): dims = [dims]
    dims = tuple(d if d >= 0 else d + data.ndim for d in dims)
    data, (iter_shape, dim_lens) = unfold_dims(data, dims, reverse=True, return_shapes=True)
    lags = [correlation_lags(l, l) for l in dim_lens]

    corr = [np.ones([l.size for l in lags])]
    for i, s in enumerate(tqdm(data[1:])):
        s1 = s
        s2 = data[i]
        if normalize:
            s1 = (s1 - np.nanmean(s1)) / np.nanstd(s1)
            s2 = (s2 - np.nanmean(s2)) / np.nanstd(s2)
        cr = correlate(s1, s2, **kwargs_correlate) / min(np.size(s1), np.size(s2))
        corr.append(cr)
    corr = np.asarray(corr)
    corr_shape = iter_shape + (*corr.shape[-len(dims):],)
    corr = corr.reshape(corr_shape)
    return lags, corr

def get_shifts_autocorrelate(data: NDArray, dims: Sequence[int] | int = -1,
                             normalize: bool = True, 
                             cumlative: bool = True,
                             kwargs_correlate: dict | None = None
                             ) -> tuple[list[NDArray], NDArray, NDArray]:
    """Estimate shifts from autocorrelation peaks.

    Parameters
    ----------
    data : NDArray
        Input signal array.
    dims : Sequence[int] | int, optional
        Dimension(s) to correlate along, by default -1.
    normalize : bool, optional
        Perform normalized cross-correlation, by default True.
    cumlative: bool, optional
        Treat the shift as a cumulative shift from the first frame, by default
        True.
    kwargs_correlate : dict, optional
        Keyword arguments passed to ``scipy.signal.correlate``, by default None.

    Returns
    -------
    lags : list[NDArray]
        Lag arrays for each dimension.
    corr : NDArray
        Autocorrelation coefficients.
    shifts : NDArray
        Shift of the data.

    See Also
    --------
    autocorrelate : Compute autocorrelation coefficients.
    maxND : Find maximum values in specified dimensions.
    shift : Shift array along dimensions using interpolation.
    """
    kwargs_correlate = {} if kwargs_correlate is None else dict(kwargs_correlate)
    lags, corr = autocorrelate(data, dims=dims,
                               normalize=normalize,
                               kwargs_correlate=kwargs_correlate)
    shifts = maxND(corr, dims=dims)[...,:-1]
    shifts[0,:] = 0
    if cumlative: shifts = np.cumsum(shifts, axis=0)
    
    return lags, corr, shifts
