
"""Interactive plotting helpers for SEA-eco widgets."""



#Imports: Typing
from __future__ import annotations
from typing import List, Callable
from numpy.typing import NDArray
from pySEA.sea_eco.architecture.base_structure_numpy import Signal

#Imports: External
from numpy import take, sum

def get_nav_plot_data(
    signal: Signal,
    function_dimensions: List[int],
    index_dimensions: List[int],
    index_values: List[int],
    nav_fnc: Callable = sum
) -> NDArray:
    """
    Apply a function along specified dimensions (expensive), take specific indices from other dimensions,
    and reorder the remaining dimensions.
    
    Parameters:
    -----------
    signal : NDArray
        Input array with arbitrary dimensions
    function_dimensions : List[int]
        List of dimension indices where the function should be applied
    index_dimensions : List[int]
        List of dimension indices from which to take specific index values
    index_values : List[int]
        List of index values corresponding to each dimension in index_dimensions
    reorder_dimensions : List[int]
        List specifying the desired order of the remaining dimensions
    nav_fnc : Callable, optional
        Function to apply along function_dimensions (default: sum)
    
    Returns:
    --------
    NDArray
        Transformed array
    
    Raises:
    -------
    ValueError
        If there are inconsistencies in the input parameters
    """
    # basic validation of original-dimension references
    if len(index_dimensions) != len(index_values):
        raise ValueError("index_dimensions and index_values must have the same length")
    all_dims = set(function_dimensions + index_dimensions)
    if not all(0 <= d < signal.dimensions.ndim for d in all_dims):
        raise ValueError(f"All dimension indices must be between 0 and {signal.dimensions.ndim-1}")
    if set(function_dimensions) & set(index_dimensions):
        raise ValueError("function_dimensions and index_dimensions cannot overlap")

    # Map the original dimensions incices and names so that as dimensionality is reduced it remains clear what axes should be referenced.
    og_mapping = signal.dimensions.get_names()
    function_dimensions = tuple([dim if isinstance(dim, str) else og_mapping[dim] for dim in function_dimensions])
    index_dimensions =    tuple([dim if isinstance(dim, str) else og_mapping[dim] for dim in index_dimensions])
    # Step 1: expensive reduction using the function
    arr = nav_fnc(signal, axis=function_dimensions) #HACK this should be dims=function_dimensions and need to add a check allowed kwarg in ufunc
    # Step 2: take indices on remaining (possibly reduced) array
    if index_dimensions:
        for dn, dv in zip(index_dimensions, index_values):
            di = arr.dimensions.get_names().index(dn)
            arr = take(arr, axis=di, indices=dv)
    return arr
