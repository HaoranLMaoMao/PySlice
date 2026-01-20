"""Crystalography-related calculator utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def bragg_reflection(hkl: int | Iterable[int],
                     lattice: float,
                     units_in: str = 'A',
                     units_out: str = '1/A',
                     twoPi: bool = True,
                     printout: bool = True) -> float:
    """Compute the Bragg reflection vector magnitude for a given Miller index.

    Parameters
    ----------
    hkl : int | Iterable[int]
        Either a pre-computed ``h^2 + k^2 + l^2`` integer or a 3-element
        iterable of Miller indices.
    lattice : float
        Lattice constant in the supplied ``units_in``.
    units_in : str, optional
        Units of ``lattice`` ('A' or 'nm'), by default 'A'.
    units_out : str, optional
        Units of the returned scattering vector ('1/A' or '1/nm'),
        by default '1/A'.
    twoPi : bool, optional
        Multiply by ``2π`` to return reciprocal-space magnitude, by default
        True.
    printout : bool, optional
        Print the d-spacing and q-value, by default True.

    Returns
    -------
    float
        Scattering vector magnitude in ``units_out``.
    """
    if isinstance(hkl, int):
        N = hkl
    elif isinstance(hkl, Iterable):
        hkl_list = list(hkl)
        if len(hkl_list) != 3:
            raise ValueError('hkl should be an integer N or an iterable with length three.')
        N = hkl_list[0]**2 + hkl_list[1]**2 + hkl_list[2]**2
    else:
        raise ValueError('hkl should be an integer N or a 3-element iterable.')
        
    if units_in=='A':
        r_scale = 1E-10
    elif units_in=='nm':
        r_scale = 1E-9
    else:
        raise ValueError("units_in must be 'A' or 'nm'.")
        
    if units_out=='1/A':
        q_scale = 1E-10
    elif units_out=='1/nm':
        q_scale = 1E-9
    else:
        raise ValueError("units_out must be '1/A' or '1/nm'.")

    d = lattice/np.sqrt(N) * r_scale
    q = 1/d * q_scale
    
    if twoPi:
        q = 2*np.pi * q

    if printout:
        print('d: {:.2f} {}'.format(d/r_scale ,units_in))
        print('q: {:.2f} {}'.format(q, units_out))
    return float(q)

