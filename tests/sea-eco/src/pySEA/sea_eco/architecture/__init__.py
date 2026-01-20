"""Architecture helpers for SEA-eco objects and decorators."""
#Imports: Typing
from __future__ import annotations
from typing import Any, Callable, List, Sequence

#Imports: External
from functools import wraps
from inspect import signature
from warnings import warn

import numpy as np


def signal_method(func: Callable) -> Callable:
    """Decorator to wrap array functions as Signal methods.

    Parameters
    ----------
    func : Callable
        Function that operates on ``self.data`` first, followed by any
        additional arguments.

    Returns
    -------
    Callable
        Wrapped function that preserves dimensions metadata when applicable.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract data from Signal args
        processed_args = []
        for arg in args:
            if hasattr(arg, 'data'): #? and hasattr(arg, 'dimensions'): Not sure if this second check is necessary.
                processed_args.append(arg.data)
            else:
                processed_args.append(arg)

        # Handle dimensions kwargs and changes
        dim_key = check_dimensions_call(func, kwargs)
        if dim_key:
            kwargs[dim_key] = self.dimensions.get_dims_as_int(kwargs[dim_key])
            if isinstance(kwargs[dim_key], List): kwargs[dim_key] = tuple(kwargs[dim_key])
            if isinstance(kwargs[dim_key], int | str | Sequence): dims_to_remove = np.atleast_1d(kwargs[dim_key])
            else: raise KeyError(f'{dim_key} takes int, str, or an Iterable and {kwargs[dim_key]} was provided.')
            remaining_dims = {i: dim for i, dim in enumerate(self.dimensions.dimensions) 
                              if i not in dims_to_remove}
        
        # Call the underlying function
        result = func(self.data, *processed_args, **kwargs)
    
        # Wrap it in a Signal
        if np.ndim(result) == self.data.ndim:
            wrapped = self.deepcopy_with_new_data(result)
        elif np.ndim(result) < self.data.ndim:
            wrapped = self.deepcopy_with_reduced_data_dim(data=result, keep_dim=remaining_dims)
        else:
            wrapped = result
        return wrapped
    
    return wrapper

def check_dimensions_call(fnc: Callable, kwargs: dict[str, Any]) -> str | None: #BUG this is only passing kwargs which then misses positional args. This needs to check positionals also then provided_key needs to include them.
    """Normalize dimension kwargs to match a function signature.
    
    Parameters
    ----------
    fnc : callable
        Function whose signature to inspect.
    kwargs : dict
        Keyword arguments supplied to ``fnc`` that may contain dimension keys.
        
    Returns
    -------
    str | None
        The dimension keyword applied to ``kwargs`` or ``None`` if no dimension
        keys were found.
    """
    possible_dim_keys = ['axis', 'axes', 'dim']

    # First check what dimension arguments the function accepts
    params = signature(fnc).parameters
    fnc_accepts = None
    for k in possible_dim_keys:
        if k in params:
            fnc_accepts = k
            break
    if fnc_accepts is None:
        warn('The function does not accept any dimensional references.')
    # Check which dimension arguments were provided
    provided_key = [k for k in possible_dim_keys if k in kwargs]

    if len(provided_key) == 0:
        return None
    elif len(provided_key) == 1:
        kwargs[fnc_accepts] = kwargs.pop(provided_key[0])
        return fnc_accepts
    elif len(provided_key) > 1:
        raise ValueError(f"Only one of {possible_dim_keys} can be used at a time, but got: {provided_key}")
        
    kwargs[fnc_accepts] = kwargs.pop(provided_key)
        
    return fnc_accepts #provided_key #? should I provide the one that was found in fnc or provided

    # if prov_dim_key: kwargs[fnc_dim_key] = kwargs.pop(prov_dim_key)
    # return prov_dim_key
