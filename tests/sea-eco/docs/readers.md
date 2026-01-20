# Readers and I/O

SEA-eco provides lightweight readers/writers for SEA HDF5 files and helpers for
third-party formats (notably Swift). Core functions live in `pySEA.sea_eco.io`.

## SEA HDF5 (`*.sea`)

- Any `SEASerializable` object (`Signal`, `SignalSet`, `AcquisitionSet`, etc.)
  can be persisted via `.to_sea("file.sea")`.
- Reload with `.from_sea("file.sea")` on an empty instance of the same class.
- Groups are tagged with `sea_type` to preserve object identity; attributes and
  datasets mirror the in-memory structure.

```python
from pySEA.sea_eco.architecture.base_structure_numpy import Signal

sig = Signal(...)  # build as usual
sig.to_sea("example.sea")

loaded = Signal()
loaded.from_sea("example.sea")
```

## Swift / third-party ingest (`pySEA.sea_eco.io`)

- `collect_swift_file(path)`: read Swift-generated `.npy`, `.ndata1/.ndata`,
  or `.h5`, returning `(metadata_dict, data_memmap)`.
- `swift_to_sea_metadata(swift_metadata, signal_type=..., ...)`: convert Swift
  metadata into `GeneralMetadata`, mapping instrument/detector/scan info into
  SEA’s schema.
- `load_memmap_from_npz`: convenience for reading arrays inside `npz` archives
  as read-only memmaps.
- `parse_file_path`: small helper that splits path/stem/extension and infers
  extension for `n*` patterns.

Example ingest flow:

```python
from pySEA.sea_eco import io
from pySEA.sea_eco.architecture.base_structure_numpy import Signal, Dimensions, Dimension
from pySEA.sea_eco.architecture.base_structure_numpy import GeneralMetadata

meta_dict, data = io.collect_swift_file("path/to/acquisition.ndata1")

# Build dimensions and metadata (simplified)
dim = Dimension(name="x", size=data.shape[1], units="px", scale=1.0, offset=0.0)
swift_meta = GeneralMetadata()
swift_meta.metadata = meta_dict  # or populate via conversion helpers
metadata = io.swift_to_sea_metadata(swift_meta, signal_type="Image")

sig = Signal(data=data, dimensions=Dimensions([dim]), metadata=metadata, name="swift import")
sig.to_sea("swift_import.sea")
```

See `examples/ThrirdPartyReaders/Nion.ipynb` for a notebook-based demonstration.
