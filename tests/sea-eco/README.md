# SEA-eco (`pySEA.sea_eco`)

pySEA (python Simulation Experiment Analysis) eco-system acrcitecture and base functionality. SEA-eco defines the data structures, metadata handling, and plotting/signal-processing utilities that live in the `pySEA` (Python Simulation Experiment Analysis) namespace package.

## Mission

- Keep SEA data **Findable, Accessible, Interoperable, and Reusable (FAIR)** from acquisition through publication. [FAIR org](https://www.go-fair.org/fair-principles/)
- Center comparisons: every signal carries calibrated dimensions and provenance so datasets can be comared.
- Stay lightweight: composable primitives inside the shared `pySEA` namespace (`from pySEA import sea_eco`).

More detail is available in `docs/mission_and_architecture.md`.

## Key features

- SEA HDF5 serialization (`.sea` files) for signals, dimensions, and metadata via `SEASerializable`.
- Calibrated `Dimension` and `Dimensions` objects for navigation/signal axis tracking.
- `Signal`, `SignalSet`, and `AcquisitionSet` containers with plotting helpers.
- Signal processing mixins for correlation, shifting, and maxima search that keep metadata in sync.
- Public calculators for microscope q-to-theta conversion and crystallography Bragg reflections.

## Installation

```bash
pip install -e .
# Optional extras
pip install -e .[dev]   # testing
pip install -e .[docs]  # Sphinx + Read the Docs theme
```

## Quick start

```python
import numpy as np
from pySEA.sea_eco.architecture.base_structure_numpy import Signal, Dimensions, Dimension

x = Dimension(name="energy", size=1024, units="eV", scale=0.1, offset=0.0)
sig = Signal(data=np.random.random(1024), dimensions=Dimensions([x]), name="demo")

sig.to_sea("demo.sea")   # persist to SEA HDF5
sig.show()               # quick plot with calibrated axes
```

## Documentation

- Sphinx configuration lives in `docs/conf.py` and uses the Read the Docs theme.
- Guides include getting started, architecture details, signal processing, plotting,
  colors, and readers (see `docs/index.rst`).
- Build locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

- Guides and architecture notes: see `docs/getting_started.md` and `docs/mission_and_architecture.md`.

## Notes on the `pySEA` namespace

SEA-eco is distributed as part of the `pySEA` namespace package. Imports follow
the pattern `from pySEA import sea_eco` or `from pySEA.sea_eco.architecture import Signal`.
