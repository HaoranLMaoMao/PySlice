"""Spike removal helpers for SEA-eco signals.

Modified from https://gist.github.com/thomasaarholt/f85ef8360682a256de260d0343d77f83#file-spikeremoval-py
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


def median_from_neighbors(energy_slice: NDArray, x: int, y: int | None = None) -> float:
    """Return the median of neighboring cells around a point.

    Parameters
    ----------
    energy_slice : NDArray
        1D or 2D array representing a line profile or map.
    x : int
        X-index of the cell.
    y : int | None, optional
        Y-index for 2D maps; ``None`` for 1D signals, by default None.

    Returns
    -------
    float
        Median of the neighboring values.
    """
    return float(np.median(find_cell_neighbors(energy_slice, x, y)))


def find_cell_neighbors(data: NDArray, X: int, Y: int | None, r: int = 1) -> list[Any]:
    """Collect neighbor values around a coordinate, excluding the center cell.

    Parameters
    ----------
    data : NDArray
        Input 1D or 2D array.
    X : int
        X-index of the target cell.
    Y : int | None
        Y-index of the target cell; ``None`` for 1D data.
    r : int, optional
        Radius (in indices) to search around the target cell, by default 1.

    Returns
    -------
    list
        Neighboring values within the specified radius.
    """
    adjacent: list[Any] = []

    if Y is None or Y == "":
        # Line spectrum
        for x in range(X - r, X + r + 1):
            if (x == X): # Do not include the spike itself
                pass
            elif (x < 0) or (x >= data.shape[0]): # Do not include indices outside the edges
                pass
            else:
                adjacent.append(data[x])
    else:
        # map
        for x in range(X - r, X + r + 1):
            for y in range(Y - r, Y + r + 1):
                if (x == X) and (y == Y): # Do not include the spike itself
                    pass
                elif (x < 0) or (x >= data.shape[0]): # Do not include indices outside the edges
                    pass
                elif (y < 0) or (y >= data.shape[1]): # Do not include indices outside the edges
                    pass
                else:
                    adjacent.append(data[x,y])
    return(adjacent)


def plot_spike_histogram(diff: NDArray, diff_after: NDArray) -> None:
    """Plot histograms of data derivatives before and after spike removal."""
    import matplotlib.pyplot as plt

    figure = plt.figure()
    hist_before, bins_before = np.histogram(diff, bins="auto")
    hist_after, bins_after = np.histogram(diff_after, bins="auto")
    center_before = (bins_before[:-1] + bins_before[1:]) / 2
    center_after = (bins_after[:-1] + bins_after[1:]) / 2
    width = 5.0 # Width of column

    plt.bar(center_before, hist_before, width=width, color = "red", label='before', log=True)
    plt.bar(center_after, hist_after, width=width, color = "blue", label='after', log=True)

    ax = plt.gca()
    (ymin, ymax) = ax.get_ylim()
    ax.set_ylim([1e-1,ymax])
    plt.legend()
    plt.title("Histogram of signal derivative")
    plt.xlabel("Derivative Magnitude")
    plt.ylabel("Counts")
    return


def remove_spikes(s: Any = None, nMAD: float = 8, plot_difference: bool = False):
    """Remove spikes that exceed a multiple of the median absolute deviation.

    Parameters
    ----------
    s : Any, optional
        Signal-like object with ``data`` and ``deepcopy`` attributes.
    nMAD : float, optional
        Threshold multiplier for MAD of the derivative, by default 8.
    plot_difference : bool, optional
        Plot derivative histograms before and after removal, by default False.

    Returns
    -------
    Any
        Deep copy of the signal with spikes replaced by neighborhood medians.

    Raises
    ------
    AttributeError
        If the signal is 1D and lacks a navigation dimension.
    """

    def mad(data: NDArray) -> float:
        """Calculate Median Absolute Deviation."""
        return float(np.median(np.abs(data - np.median(data))))

    shape = s.data.shape
    if len(shape) == 1:
        # Case single spectrum
        raise AttributeError("Remove spikes requires a Spectrum Image with non-zero navigation dimension")

    elif len(shape) == 2:
        # Case Line profile
        print("Recognised as Line Profile")
        diff = np.diff(s.data, axis=0)
        threshold = nMAD * mad(diff.flatten()) # MAD - nMad is the number of deviations away from the median are included
        print("Gradient threshold is " + str(threshold))

        spike_positions: list[Sequence[int]] = []

        positive = diff > threshold # Position of any inclining spikes
        (x,e) = np.nonzero(positive) # Get index of spikes
        x += 1 # Position of spike is the array index ahead of the gradient
        for i in range(len(x)):
            spike_positions.append([x[i], e[i]])

        negative = diff < -threshold # Position of any declining spikes
        (x,e) = np.nonzero(negative) # Get index of spikes
        for i in range(len(x)):
            spike_positions.append([x[i], e[i]])

        print("Found " + str(len(spike_positions)) + " spikes!")

        for (x,e) in spike_positions:
            s.data[x,e] = median_from_neighbors(s.data[:,e], x)

        if plot_difference == True:
            diff_after = np.diff(s.data, axis=0)
            plot_spike_histogram(diff, diff_after)

    elif len(shape) == 3:
        # Case EELS Map
        print("Recognised as 2D Spectrum Image")
        diff = np.diff(s.data, axis=1) # Get differential across data
        threshold = nMAD * mad(diff.flatten()) # MAD
        print("Gradient threshold is " + str(threshold))

        positive = diff > threshold # Position of any inclining spikes
        negative = diff < -threshold # Position of any declining spikes

        spike_positions: list[Sequence[int]] = []

        (x,y,e) = np.nonzero(positive) # Get index of spikes
        y += 1 # Position of spike is the array index ahead of the gradient
        for i in range(len(x)):
            spike_positions.append([x[i], y[i], e[i]])

        (x,y,e) = np.nonzero(negative) # Get index of spikes
        for i in range(len(x)):
            spike_positions.append([x[i], y[i], e[i]])

        print("Found " + str(len(spike_positions)) + " spikes!")

        for (x,y,e) in spike_positions:
            # Spike intensity replaced by median of neighbors
            s.data[x,y,e] = median_from_neighbors(s.data[:,:,e], x,y)

        if plot_difference == True:
            diff_after = np.diff(s.data, axis=1)
            plot_spike_histogram(diff, diff_after)
    else:
        print("The signal shape does not match an expected value (X, S or X, Y, S). Signal data shape is " + str(s.data.shape))

    return s.deepcopy()
