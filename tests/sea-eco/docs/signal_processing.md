# Signal Processing Guide

Signal processing utilities live primarily in
`pySEA.sea_eco.mixins.signal_mixins.SignalProcessingMixin` and the
`pySEA.sea_eco._signal_processing` module. The mixin is attached to
`Signal`, so you can call these methods directly on signals.

## Correlation and alignment

```python
import numpy as np
from pySEA.sea_eco.architecture.base_structure_numpy import Dimension, Dimensions, Signal

t = Dimension(name="time", size=256, units="s", scale=0.5, offset=0.0)
sig = Signal(data=np.random.random((10, 256)), dimensions=Dimensions([Dimension(name="scan", size=10, units="idx"), t]), name="frames")

# Autocorrelation along the time axis
corr_sig = sig.autocorrelate(dims=-1)

# Lag-aware shifts from autocorrelation peaks
shifts = sig.get_shifts_autocorrelate(dims=-1, cumlative=True)

# Cross-correlation against a reference
ref = sig[0]  # first frame
lags_sig = sig.correlate_1D_in_ND(ref_sig=ref, dims=-1)
```

Features:

- Dimension-aware: specify axes by index or name; outputs include calibrated lag
  dimensions when possible.
- Normalization and lag computation handled via SciPy (`scipy.signal.correlate`).
- Shifts can be cumulative or per-frame; use `cumlative=False` to get
  frame-to-frame deltas.

## Shifting data

```python
# Apply a calibrated shift along time
shifted = sig.shift(shifts=2.5, shifts_calibrated=True, dims=-1, interpolate_method="cubic")
```

- Accepts scalar or per-sample shift arrays (broadcasted to the data shape).
- Supports `'linear'` or `'cubic'` interpolation, or a custom callable.
- Preserves metadata and dimensions on the returned `Signal`.

## Maxima search

```python
peaks = sig.maxND(dims="signal")  # returns indices + max value along signal dims
```

- Returns an array where the last entries are the indices of maxima (optionally
  calibrated) followed by the maximum value.

## Lower-level primitives

For library authors or batch workflows, the raw functions in
`pySEA.sea_eco._signal_processing` are available:

- `unfold_dims`, `maxND`, `shift`
- `correlate_1D_in_ND`, `autocorrelate`, `get_shifts_autocorrelate`

See `examples/Signal_processing/correlation_and_alignment.ipynb` for a fuller
walkthrough including plotting of correlation peaks and applied shifts.
