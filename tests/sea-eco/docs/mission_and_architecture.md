# Mission and Architecture

SEA-eco lives inside the `pySEA` (Python Simulation Experiment Analysis)
namespace and provides the core Simulation Experiment Analysis data model,
plotting helpers, and signal processing utilities.

## Mission statement

- Empower reproducible Simulation Experiment Analysis by keeping data **Findable, Accessible, Interoperable, and Reusable (FAIR)** from acquisition through publication.
- Treat the `pySEA` namespace as a shared ecosystem: small, composable primitives rather than monoliths; stable serialization so signals can move between labs, clusters, and notebooks.
- Make comparison the default: every dataset travels with calibrated dimensions, provenance, and metadata so results can be aligned, re-processed, and peer-reviewed.

## Notes on the `pySEA` namespace

`sea_eco` is published under the `pySEA` namespace package [pySEA](https://github.com/sea-ecosystem). The intent of the namespace is to keep the ecosystem lightweight, modular, and built upon the same architecture for streamlined data analysis workflows.
Imports therefore take the form `from pySEA import sea_eco` or
`from pySEA.sea_eco.architecture import ...`.
Packages included in [pySEA](https://github.com/sea-ecosystem):
- [sea-eco](https://github.com/sea-ecosystem/sea-eco): The base ecosystem implementing the architecture a basic tools
- [rayTEM](https://github.com/sea-ecosystem/rayTEM): A TEM digital-twin enabling automation, infromed (live) isntrument alignment, and instrument infromed SEA objects.
- [PySlice](https://github.com/h-walk/PySlice): electron multislice simulation software with built in ML-MD universal potenetials to include realistic lattice motion.

## FAIR principles in practice
We strive to implement the principles established by [FAIR org](https://www.go-fair.org/fair-principles/)
- **Findable:** SEA files (`*.sea`) embed human-readable metadata, SEA type tags, and UUIDs for signals and acquisitions.
- **Accessible:** HDF5-backed storage keeps data portable; helper utilities expose quick tree views (`get_tree_html`, `show_tree`) for inspection.
- **Interoperable:** Calibrated `Dimension` objects and unit-aware plotting/processing make cross-instrument comparisons predictable.
- **Reusable:** Serialization/deserialization (`to_sea`, `from_sea`) preserve dimensions, metadata, and relationships between signals and sets.

## Architectural overview (numpy backend)

- **SEASerializable:** Base class that provides `.to_dict()`, `.to_sea()`, and `.from_hdf5_group()` for lossless round-tripping to SEA-formatted HDF5.
- **Metadata:** Nested, mergeable metadata container that keeps provenance close to the data.
- **Dimension / Dimensions:** Calibrated axes with names, units, scales, offsets, and navigation/signal role tracking.
- **Signal:** N-dimensional array + `Dimensions` + `Metadata`, with plotting helpers and numpy-like ufunc dispatch via decorators.
- **SignalProcessingMixin:** Adds basic signal processing utilities (correlation, shifts, maxima search) that keep dimension metadata synchronized.
- **SignalCollection:** Container for Signals and SignalSets with shared metadata.
- **SignalSet:** Group related signals with shared dimensions; can decompose/merge detector metadata.
- **SEAFile:** Top-level container mirroring a `.sea` file with Simulations/Experiments/Analysis collections and global metadata.

## Architecture for data comparison

- **Dimension-first alignment:** Comparisons run through `Dimensions.get_dims_as_int` so operations can refer to axes by name or index.
- **Calibrated shifting/correlation:** `SignalProcessingMixin` exposes shift/correlation helpers that return calibrated lag dimensions for inspection and downstream use.
- **SEA file provenance:** Every persisted object records `sea_type` and UUIDs, allowing heterogeneous datasets to be compared while retaining lineage.
- **Tree introspection:** HTML and string tree renderers make it simple to audit an object's structure before combining or comparing datasets.
