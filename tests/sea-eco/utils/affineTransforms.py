"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely DPC.
"""
import numpy as np
try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    xp = np



def rotation_matrix(angle:float, degrees:bool=True):
    '''Rotation matrix
    Paramaters
    angle: float
    Angle in degrees or radian
    degrees: bool
    Angle is given in degrees.
    '''
    if degrees: angle = np.deg2rad(angle)
    return xp.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])