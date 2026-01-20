"""Color-wheel utilities for 2D hue/saturation plotting."""

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb
from numpy.typing import NDArray


def get_color_wheel(N: int = 301, phi: float = 0,
                    vmax: float | None = None, clear: bool = True) -> NDArray:
    """Generate an RGBA color wheel image.

    Parameters
    ----------
    N : int, optional
        Resolution of the generated square image. Default is 301.
    phi : float, optional
        Rotation angle in degrees applied to the hue. Default is 0.
    vmax : float | None, optional
        Maximum radius for normalization; when ``None`` uses 1. Default is
        None.
    clear : bool, optional
        If True, masks pixels outside the unit circle (alpha set to 0).
        Default is True.

    Returns
    -------
    NDArray
        RGBA image of shape ``(N, N, 4)`` representing the color wheel.
    """
    r = 1 if vmax is None else vmax
    x = np.linspace(-r, r, N)
    x, y = np.meshgrid(x, x, indexing='ij')
    phi = np.deg2rad(phi)
    
    comp = x + 1j*y
    hsv = np.stack([(np.angle(comp)+phi)/(2*np.pi)%1,
                    np.ones_like(x),
                    np.abs(comp)/r], axis=-1)
    rgb = hsv_to_rgb(hsv)
    rgbt = np.concatenate((rgb, np.ones(x.shape+(1,))), axis=-1)
    if clear:
        mask = np.any(rgbt[...,:3]>1, axis=-1)
        rgbt[mask, -1] = 0
    
    return rgbt

def plot_color_wheel(ax: Axes, N: int = 301, phi: float = 0,
                     vmax: float | None = None, clear: bool = True) -> None:
    """Render a color wheel into an axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render into.
    N : int, optional
        Resolution of the generated square image. Default is 301.
    phi : float, optional
        Rotation angle in degrees applied to the hue. Default is 0.
    vmax : float | None, optional
        Maximum radius for normalization; when ``None`` uses 1. Default is
        None.
    clear : bool, optional
        If True, masks pixels outside the unit circle. Default is True.
    """
    rgbt = get_color_wheel(N=N, phi=phi, vmax=vmax, clear=clear)
    ax.imshow(rgbt, extent=[-1,1,-1,1], origin='lower')

def get_color_hexagon(N: int = 301, phi: float = 0, clear: bool = True) -> NDArray:
    """Generate a hexagonal cutout of the color wheel.

    Parameters
    ----------
    N : int, optional
        Resolution of the generated square image. Default is 301.
    phi : float, optional
        Rotation angle in degrees applied to the hue. Default is 0.
    clear : bool, optional
        If True, masks pixels outside the hexagon. Default is True.

    Returns
    -------
    NDArray
        RGBA image of shape ``(N, N, 4)`` with a hexagonal mask applied.
    """
    rgbt = get_color_wheel(N=N, phi=phi, clear=True)
    x = np.linspace(-1,1,N)
    x,y = np.meshgrid(x,x, indexing='xy')
    
    mask_side = np.abs(x)>np.cos(np.deg2rad(30))
    rgbt[mask_side,-1] = 0
    mask_top = np.abs(y) > 1-np.abs(x)*np.tan(np.deg2rad(30))
    rgbt[mask_top, -1] = 0
    
    return rgbt

def plot_color_hexagon(ax: Axes, N: int = 301, phi: float = 0,
                       clear: bool = True,
                       labels: Sequence[str] | None = None,
                       labelpad: float = 0.1,
                       font_kwargs: Mapping[str, object] | None = None) -> None:
    """Render a hexagonal color wheel into an axes with optional vertex labels.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render into.
    N : int, optional
        Resolution of the generated square image. Default is 301.
    phi : float, optional
        Rotation angle in degrees applied to the hue. Default is 0.
    clear : bool, optional
        If True, masks pixels outside the hexagon. Default is True.
    labels : Sequence[str] | None, optional
        Optional three labels placed at the hexagon vertices. Default is None.
    labelpad : float, optional
        Padding applied to the label positions. Default is 0.1.
    font_kwargs : Mapping[str, object] | None, optional
        Additional keyword arguments passed to ``ax.text`` for labeling.
        Default is None.
    """
    font_kwargs = {} if font_kwargs is None else dict(font_kwargs)
    rgbt = get_color_hexagon(N=N, phi=phi, clear=clear)
    ax.imshow(rgbt, extent=[-1,1,-1,1], origin='lower')
    if labels is not None and len(labels) == 3:
        deg = np.deg2rad(30)
        ax.text(0, 1-labelpad, labels[0], ha='center', va='center', **font_kwargs)
        ax.text(np.cos(deg)*(1-labelpad), (labelpad-1)*np.sin(deg), labels[1], ha='center', va='center', **font_kwargs)
        ax.text(-np.cos(deg)*(1-labelpad), (labelpad-1)*np.sin(deg), labels[2], ha='center', va='center', **font_kwargs)

def plot_rgb_traingle(ax: Axes, N: int = 301,
                      labels: Sequence[str] | None = None,
                      labelpad: float = 0.1,
                      font_kwargs: Mapping[str, object] | None = None,
                      scheme: str = 'rgb') -> None:
    """Plot a Maxwell triangle with primary/secondary labels.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render into.
    N : int, optional
        Resolution of the generated square image. Default is 301.
    labels : Sequence[str] | None, optional
        Optional three labels placed at the triangle vertices. Default is None.
    labelpad : float, optional
        Padding applied to the label positions. Default is 0.1.
    font_kwargs : Mapping[str, object] | None, optional
        Additional keyword arguments passed to ``ax.text``. Default is None.
    scheme : str, optional
        Color scheme, either ``'rgb'`` or ``'cmy'`` (complementary). Default is
        'rgb'.
    """
    font_kwargs = {} if font_kwargs is None else dict(font_kwargs)
    img = np.zeros((N,N,4))
    dx = 2.0/N
    dy = 1.0/N
    x = np.linspace(-1,1,N)
    y = np.linspace(0,1,N)

    x,y = np.meshgrid(x,y, indexing='ij')

    r = y
    g = (x+1-r)/2
    b = 1.0-g-r
    t = np.zeros_like(r)
    rgbt = np.stack((r,g,b,t)).T
    if scheme == 'cmy': rgbt = [1,1,1,0] - rgbt

    mask1 = np.all(rgbt[...,:3]>=0, axis=-1)
    mask2 = np.all(rgbt[...,:3]<=1, axis=-1)
    mask = np.logical_and(mask1, mask2)
    rgbt[mask, -1] = 1
    
    a = 1.0/np.sqrt(3)
    ax.imshow(rgbt, origin='lower',extent=[-a,a,-1/3,2/3])
    ax.set_aspect('equal')
    
    if labels is not None and len(labels) == 3:
        deg = np.deg2rad(30)
        ax.text(0, 2/3-labelpad, labels[0], ha='center', va='center', **font_kwargs)
        ax.text(a - np.cos(deg)*labelpad, -1/3 + np.sin(deg)*labelpad, labels[1], ha='center', va='center', **font_kwargs)
        ax.text(-a + np.cos(deg)*labelpad, -1/3 + np.sin(deg)*labelpad, labels[2], ha='center', va='center', **font_kwargs)
