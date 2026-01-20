# Plotting Signals

SEA-eco ships with plotting helpers that respect calibrated dimensions and
metadata. These utilities live in `pySEA.sea_eco._plotting.plot` and are exposed
via `Signal.show()` and `Signal.image()`.

```python
import numpy as np
from pySEA.sea_eco.architecture.base_structure_numpy import Dimension, Dimensions, Signal

x = Dimension(name="x", size=128, units="nm", scale=0.5, offset=0.0)
y = Dimension(name="y", size=128, units="nm", scale=0.5, offset=0.0)
sig = Signal(data=np.random.random((128, 128)),
             dimensions=Dimensions([y, x]),
             name="image")

# Quick static plot (infers best dimensions to show)
p = sig.show(cmap="gray", scale_bar_kwargs={"units": "nm"})

# Persist a rendered image with scale bar metadata
sig.image(filename="image.tif")
```

## Dimension inference

- If no dims are passed, `Signal.show` will:
  - Use the two signal dimensions if <= 2 exist.
  - Otherwise fall back to navigation dimensions when <= 2.
- Use the `dims` argument to force `('sig' | 'nav' | iterable of axes)` and the
  optional `fnc` argument (default `np.sum`) to reduce other axes.

## Plot utilities

The underlying helpers live in `pySEA.sea_eco._plotting.plot`:

- `plot_nd_array` renders N-D arrays with sensible default extents.
- `save_fig` and `save_image` save matplotlib figures or raw arrays alongside
  calibration metadata.
- `PlotImage` is a thin wrapper that stores handles for additional overlays.

## Interactive plotting

An interactive widget (`interactive_signal_plot`) exists in
`pySEA.sea_eco._plotting.interactive.ipywidget`. It is currently commented out
in `Signal.show_interactive` due to a circular import but can be used directly
in notebooks when the widget stack is available (`ipympl`, `ipywidgets`).
