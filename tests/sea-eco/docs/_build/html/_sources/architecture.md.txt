# SEA-eco Architecture (numpy backend)

This page expands on the `pySEA.sea_eco.architecture.base_structure_numpy`
module. For a runnable walkthrough see `examples/Basic usage.ipynb`.

## Core building blocks

- **SEASerializable**: Base mixin that provides `.to_dict()`, `.to_sea()`, and
  `.from_hdf5_group()` for lossless round-tripping to SEA-formatted HDF5.
- **Metadata**: Nested, mergeable metadata container that keeps provenance close
  to the data.
- **Dimension**: A calibrated axis with `name`, `size`, `units`, `scale`,
  `offset`, optional `values`, and flags for navigation/signal roles.
- **Dimensions**: Ordered collection of `Dimension` objects with helpers to:
  - Map between names/indices (`get_index_from_name`, `get_dims_as_int`)
  - Track navigation vs signal dimensions (`nav_dimensions`, `sig_dimensions`)
  - Manage extents and calibrated values (`get_extent`, `get_extents`)
  - Add/remove dimensions while keeping role bookkeeping in sync.
- **Signal**: N-dimensional array + `Dimensions` + `Metadata`:
  - NumPy ufunc passthrough that preserves metadata/dimensions
  - HDF5 serialization via the SEA format (`to_sea`, `from_sea`)
  - Plotting helpers (`show`, `image`) that infer sensible axes
  - Dimension-aware reshape utilities (`unfold_axes`, `fold_axes`)
  - Child factory (`generate_child_signal`) to keep lineage (`uuid`) intact.
- **SignalCollection**: Container for Signals and SignalSets with shared
  metadata; supports add/remove, SEA serialization, and metadata merging.
- **SignalSet**: Specialized collection that enforces shared dimensions across
  signals and can decompose/merge detector metadata.
- **SEAFile**: Top-level container mirroring a `.sea` file; bundles
  `Simulations`, `Experiments`, and `Analysis` collections plus global metadata.

## SEA HDF5 layout (``*.sea``)

- Each SEA object serializes into an HDF5 group tagged with `sea_type`.
- Attributes are stored as HDF5 attributes when scalar/short; arrays use datasets.
- Nested SEA objects (signals, dimensions, metadata) become child groups,
  preserving structure and traversal order.
- UUIDs are generated via `generate_uuid()` and embedded alongside objects to
  support provenance and comparisons across files.

## Useful patterns (mirrors “Basic usage.ipynb”)

```python
import numpy as np
from pySEA.sea_eco.architecture.base_structure_numpy import (
    Dimension, Dimensions, Signal, SignalSet, SEAFile
)

# 1) Build a calibrated signal
energy = Dimension(name="energy", size=2048, units="eV", scale=0.1, offset=0.0)
sig = Signal(data=np.random.random(2048),
             dimensions=Dimensions([energy]),
             name="EELS spectrum")

# 2) Duplicate signals while keeping lineage
denoised = sig.deepcopy_with_new_data(data=np.convolve(sig.data, np.ones(5)/5, mode="same"))
denoised.name = "EELS spectrum (smoothed)"

# 3) Bundle signals for comparison
sig_set = SignalSet(signals=[sig, denoised], main_signal=0, merge_dimensions=True)

# 4) Persist and reload via SEA HDF5
sea_file = SEAFile(experiments=[sig_set])
sea_file.to_sea("bundle.sea")
reloaded = SEAFile()
reloaded.from_sea("bundle.sea")
```

## Dimension-aware comparisons

- Use `Dimensions.get_dims_as_int` to allow API calls by name or index.
- `Signal.deepcopy_with_reduced_data_dim` keeps navigation/signal role
  bookkeeping in sync when projections reduce dimensionality.
- Tree inspection (`get_tree_html`, `show_tree`) is helpful before merging or
  comparing datasets from different instruments.
