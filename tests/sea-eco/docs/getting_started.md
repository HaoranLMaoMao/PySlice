# Getting Started

SEA-eco is distributed as part of the `pySEA` (Python Simulation Experiment
Analysis) namespace package. Install it in
editable mode while developing or exploring:

```bash
pip install -e .
```

Optional extras for building docs and running examples:

- `pip install -e .[docs]` for the Sphinx toolchain and the Read the Docs theme
- `pip install -e .[dev]` for testing utilities

## Quick usage

```python
import numpy as np
from pySEA import sea_eco

# Build a simple 1D signal with calibrated dimensions
from pySEA.sea_eco.architecture.base_structure_numpy import Signal, Dimensions, Dimension

x = Dimension(name="energy", size=1024, units="eV", scale=0.1, offset=0.0)
signal = Signal(data=np.random.random(1024), dimensions=Dimensions([x]), name="demo")

# Serialize to the SEA HDF5 format
signal.to_sea("demo.sea")

# Plot or process the signal
_ = signal.show()
lags, corr = sea_eco.signal_processing.autocorrelate(signal.data)
```

## Building the docs locally

```bash
cd docs
sphinx-build -b html . _build/html
```

Then open `_build/html/index.html` in your browser. The configuration lives in
`docs/conf.py` and is set up for the Read the Docs theme.
