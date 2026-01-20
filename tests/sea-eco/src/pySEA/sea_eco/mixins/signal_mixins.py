"""Signal processing mixins for SEA-eco Signal objects."""

#Imports: Typing
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Literal, Sequence
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pySEA.sea_eco.architecture.base_structure_numpy import Dimensions, Signal

#Imports: External
from functools import wraps

import numpy as np

#Imports: Internal
from pySEA.sea_eco.architecture import signal_method
from pySEA.sea_eco._signal_processing import maxND, shift, correlate_1D_in_ND, autocorrelate, get_shifts_autocorrelate

dims_set_maps = {'signal':'sig_dimensions', 'navigation':'nav_dimensions',
                 'temporal':'temporal_dimension', 'spectral':'spectral_dimension',
                 'position':'position_dimensions', 'scattering':'scattering_dimensions'}
dims_set_Lit = Literal['signal', 'navigation', 'temporal', 'spectral', 'position', 'scattering']

def get_corr_dimensions(signal: Signal, dims: Sequence[int],
                        lags: NDArray, corr: NDArray) -> Dimensions:
    """Update dimension metadata with lag offsets and sizes."""
    sig_dims = signal.dimensions.deepcopy()
    for dim,lag in zip(dims, lags):
        scale = sig_dims[dim].scale
        sig_dims[dim].offset =  scale*np.min(lag)
        sig_dims[dim].size = corr.shape[dim]
    return sig_dims

class SignalProcessingMixin:

    def maxND(self, dims: Sequence | dims_set_Lit = 'signal', 
              calibrated: bool = True
              ) -> NDArray:
        """Find the maxium value and its index in a set of dimensions.

        Parameters
        ----------
        dims : tuple, optional
            Axes to find the max within, by default (-2,-1)

        Returns
        -------
        max_values: NDArray
            Maximum values with the shape (*data.shape,len(dims)+1) with non-dims removed. The last dimension coresponds to the dims indicies and the max value.
        """
        if isinstance(dims, str):
            if dims in dims_set_maps:
                dims = tuple(getattr(self.dimensions, dims_set_maps[dims]))
            else:
                raise UserWarning(f'dims takes an {tuple[int] | dims_set_Lit}')
        dims = tuple(d if d>=0 else int(d+self.dimensions.ndim) for d in dims)

        max_info = maxND(self.data, dims=dims)
        if calibrated:
            pos = max_info[...,:-1].astype(int)
            for i, dim in enumerate(dims): pos[...,i] = self.dimensions[dim].get_calibrated_value(pos[...,i])
            max_info[...,:-1] = pos
        return max_info

    def shift(self, shifts: NDArray,
              shifts_calibrated = True,
              dims: int | Sequence[int] = -1,
              interpolate_method: Callable | Literal['linear', 'cubic'] = 'cubic',
              kwargs_interpolate: dict = {},
              ) -> Signal:
        """Shift an array along one or more dimensions using interpolation.

        Parameters
        ----------
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
        Signal
            Shifted signal.
        """
        if isinstance(dims, int): dims = [dims]
        scales = [self.dimensions[dim].scale for dim in dims]
        if shifts_calibrated:
            shifts /= scales
        data = shift(self.data, shifts,
                     dims=dims,
                     interpolate_method=interpolate_method,
                     kwargs_interpolate=kwargs_interpolate)
        
        sig = self.deepcopy_with_new_data(data)
        sig.name = f'{self.name} shifted'
        return sig

    def correlate_1D_in_ND(self, ref_sig : Signal | NDArray,
                           dims: int = -1,
                           normalize: bool = True, kwargs_correlate: dict = {}
                           ) -> Signal:
        """Perform 1D cross-correlation across a ND signal.
        If the signal has shifted positive with respect tothe reference then the correlation will return a positive lag.

        Parameters
        ----------
        ref_sig : Signal | NDArray
            Refference dataset to correlate against.
        dims : int, optional
            Dimension to correlate along. By default -1
        normalize : bool, optional
            Perfrom normalized cross-correlation. By default True.
        kwargs_correlate : dict, optional
            kwargs for scipy.signal.correlate, by default {}

        Returns
        -------
        Correlations: Signal
            Correlation coefficients.

        See Also
        --------
        scipy.signal.correlate
        """
        
        lags, corr = correlate_1D_in_ND(self.data, ref_sig.data,
                                        dims=dims,
                                        normalize=normalize, kwargs_correlate=kwargs_correlate)
        off1 =  self.dimensions[dims].offset
        scale = self.dimensions[dims].scale
        
        if hasattr(ref_sig, 'dimensions'):
            off2 = ref_sig.dimensions[dims].offset
        else: off2 = 0

        sig_dims = self.dimensions.deepcopy()
        sig_dims[dims].offset =  scale*np.min(lags) + (off2-off1)
        sig_dims[dims].size = corr.shape[dims]

        corr_sig = self.generate_child_signal(corr,
                                               name= 'Correlation coefficients',
                                               dimensions= sig_dims)
        return corr_sig
    
    def autocorrelate(self, dims: Sequence[int] | int = -1,
                       normalize: bool = True,
                       kwargs_correlate: dict = {}
                       ) -> NDArray:
        """Measure autocorrelation of a signal across specified dimensions.

        Parameters
        ----------
        dims : Sequence[int] | int, optional
            Dimension(s) to correlate along, by default -1
        normalize : bool, optional
            Perform normalized cross-correlation, by default True
        kwargs_correlate : dict, optional
            Keyword arguments passed to scipy.signal.correlate, by default {}

        Returns
        -------
        lags : list[NDArray]
            Lag arrays for each dimension.
        corr : NDArray
            Autocorrelation coefficients.
        """ 
        if isinstance(dims, int): dims = [dims]
        lags, corr = autocorrelate(self.data, dims=dims,
                                    normalize=normalize, 
                                    kwargs_correlate=kwargs_correlate)
        
        sig_dims = get_corr_dimensions(self, dims, lags, corr)

        corr_sig = self.generate_child_signal(corr,
                                               name= 'Correlation coefficients',
                                               dimensions= sig_dims)
        return corr_sig
    
    def get_shifts_autocorrelate(self, dims: Sequence[int] | int = -1,
                             normalize: bool = True, 
                             cumlative: bool = True,
                             kwargs_correlate: dict = {}
                             ) -> tuple[list[NDArray], NDArray, NDArray]:
        """Measure autocorrelated shift of a signal across specified dimensions.
        
        Parameters
        ----------
        dims : Sequence[int] | int, optional
            Dimension(s) to correlate along, by default -1
        normalize : bool, optional
            Perform normalized cross-correlation, by default True
        cumlative: bool, optional
            Treat the shift as a cumlative shift from the first frame, by default True
        kwargs_correlate : dict, optional
            Keyword arguments passed to scipy.signal.correlate, by default {}

        Returns
        -------
        shfit : NDArray
            Autocorrelation shift.
        """ 
        dims = list(d if d>=0 else int(d+self.dimensions.ndim) for d in dims)
        
        corr = self.autocorrelate(dims=dims, 
                                  normalize=normalize,
                                  kwargs_correlate=kwargs_correlate)
                
        sig_dims = self.dimensions.deepcopy()
        for d in np.sort(dims)[::-1]:
            sig_dims.remove_dimension(d)
        
        
        shifts = corr.maxND(dims=dims)[...,:-1]
        shifts[0,:] = 0
        if cumlative: shifts = np.cumsum(shifts, axis=0) #? Should this always be `axis=zero`?

        shift_sig = self.generate_child_signal(shifts,
                                    name= 'Autocorrelated shifts',
                                    dimensions= sig_dims)

        return shift_sig
    
    def get_shifts_correlate_1D_in_ND(self, ref_sig : Signal | NDArray,
                                      dims: int = -1,
                                      normalize: bool = True, 
                                      kwargs_correlate: dict | None = None,
                                      avg_corr_dims: Sequence[int] | int = [],
                                      cumlative: bool = True
                                      ) -> Signal:
        """Perform 1D cross-correlation across a ND signal.
        If the signal has shifted positive with respect tothe reference then the correlation will return a positive lag.

        Parameters
        ----------
        ref_sig : Signal | NDArray
            Refference dataset to correlate against.
        dims : int, optional
            Dimension to correlate along. By default -1
        normalize : bool, optional
            Perfrom normalized cross-correlation. By default True.
        kwargs_correlate : dict, optional
            kwargs for scipy.signal.correlate, by default None
        cumlative: bool, optional
            Treat the shift as a cumlative shift from the first frame, by default True

        Returns
        -------
        Correlations: Signal
            Correlation coefficients.

        See Also
        --------
        scipy.signal.correlate
        """
        if isinstance(avg_corr_dims, int): 
            avg_corr_dims = [avg_corr_dims]
        dim_names = self.dimensions.get_names()
        avg_corr_dims = [dim_names[i] for i in avg_corr_dims]
    
        corr = self.correlate_1D_in_ND(ref_sig, dims=dims,
                                       normalize=normalize,
                                       kwargs_correlate=kwargs_correlate
                                       )

        if len(avg_corr_dims)>0:
            avg_corr_dims = [corr.dimensions.get_index_from_name(n) for n in avg_corr_dims]
            corr = np.nanmean(corr, axis=avg_corr_dims)
        
        shifts = corr.maxND(dims=[dims])[...,0]
        if cumlative: shifts = np.cumsum(shifts, axis=0) #? Should this always be `axis=zero`?

        # shift_sig = shifts.generate_child_signal(shifts,
        #                             name= 'Correlated shifts')

        return shifts
