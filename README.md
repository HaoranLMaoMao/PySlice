# PySlice

A GPU-accelerated Python package for simulating vibrational electron energy loss spectroscopy (EELS) using the **TACAW method** (Time Autocorrelation of Auxiliary Wavefunctions). PySlice integrates molecular dynamics with multislice electron scattering calculations to predict momentum- and energy-resolved phonon spectra directly from atomic trajectories.

## Features

- **TACAW Analysis**: Convert time-domain electron scattering into frequency-domain phonon spectra
- **Integrated MD**: Run molecular dynamics with universal ML potentials (ORB, MACE, CHGNet)
- **GPU Acceleration**: PyTorch backend with automatic CUDA/MPS/CPU selection
- **Flexible Input**: Load structures from CIF, LAMMPS, XYZ, ASE trajectories, or ASE Atoms objects
- **STEM Imaging**: HAADF/ADF/BF imaging and 4D-STEM diffraction

## Installation

```bash
# Clone the repository
git clone https://github.com/h-walk/PySlice.git
cd PySlice

# Install with pip. -e = editable mode. [fast] will install torch (technically optional, but provides extreme speed improvements)
pip install -e ".[fast]"

# Install OVITO for trajectory loading
pip install ovito --find-links https://www.ovito.org/pip/

# Or using uv (recommended)
uv sync
```

## Quick Start

### Full TACAW Pipeline (MD → Multislice → Phonon Dispersion)

```python
from ase.build import bulk
from pyslice.md import MDCalculator
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.tacaw_data import TACAWData

# 1. Create structure
atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (10, 10, 2)

# 2. Run molecular dynamics
md = MDCalculator(model_name="orb-v3-direct-inf-omat")
md.setup(atoms, temperature=300, timestep=2.0, production_steps=500, save_interval=5)
trajectory = md.run()

# 3. Run multislice (parallel beam for TACAW)
calc = MultisliceCalculator()
calc.setup(trajectory, aperture=0, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5)
wf_data = calc.run()

# 4. Compute phonon spectrum
tacaw = TACAWData(wf_data)
Z = tacaw.spectral_diffraction(15.0)  # Diffraction at 15 THz
```

### Load Existing Trajectory

```python
from pyslice.io.loader import Loader

# LAMMPS dump file
trajectory = Loader(
    "trajectory.lammpstrj",
    timestep=0.01,  # ps
    atom_mapping={1: "Si", 2: "Ge"}
).load()

# ASE trajectory or CIF/XYZ file
trajectory = Loader("structure.cif").load()
```

### HAADF-STEM Imaging

```python
from pyslice.multislice.multislice import probe_grid
from pyslice.postprocessing.haadf_data import HAADFData

# Define probe scan grid
xy = probe_grid([0, 20], [0, 20], nx=32, ny=32)

calc = MultisliceCalculator()
calc.setup(trajectory, aperture=30, voltage_eV=100e3, sampling=0.1, probe_positions=xy)
wf_data = calc.run()

haadf = HAADFData(wf_data)
haadf.calculateADF(inner_mrad=60, outer_mrad=200)
haadf.plot()
```

### TEM Diffraction

```python
calc = MultisliceCalculator()
calc.setup(trajectory, aperture=0, voltage_eV=100e3, sampling=0.1)
wf_data = calc.run()
wf_data.plot(powerscaling=0.125)  # Diffraction pattern
```

## Data Flow

```
Input Sources          Processing              Analysis            Output
─────────────────────────────────────────────────────────────────────────────
CIF / XYZ / LAMMPS ─┬─→ Loader ─┬─→ MDCalculator ─┐
ASE Atoms / .traj  ─┘           │   (ORB, MACE)   │
                                │                 ↓
                                └───────────→ Trajectory
                                                  │
                                                  ↓
                                          MultisliceCalculator
                                          (Probe → Potential → Propagate)
                                                  │
                                                  ↓
                                              WFData ψ(k,t)
                                                  │
                        ┌─────────────────────────┼─────────────────────────┐
                        ↓                         ↓                         ↓
                   TACAWData                 HAADFData                  WFData
                   FFT(t)→ω                  ∫|ψ|²dΩ                  (direct)
                        │                         │                         │
                        ↓                         ↓                         ↓
                Phonon Dispersion           STEM Image              Diffraction
                Spectral Diffraction        ADF/HAADF/BF            CBED/LACBED
                Spectrum Image                                      4D-STEM
```

## Main Classes

### `Loader`
Load atomic structures and trajectories from various formats.

```python
from pyslice.io.loader import Loader

# Supported: CIF, XYZ, LAMMPS dump, ASE .traj, ASE Atoms objects
traj = Loader("file.cif").load()
traj = Loader("dump.lammpstrj", timestep=0.01, atom_mapping={1: "B", 2: "N"}).load()
```

### `MDCalculator`
Run molecular dynamics with universal ML potentials.

```python
from pyslice.md import MDCalculator

md = MDCalculator(model_name="orb-v3-direct-inf-omat", device="cuda")
md.setup(
    atoms,
    temperature=300,        # K
    timestep=2.0,           # fs
    production_steps=1000,
    save_interval=5,
)
trajectory = md.run()
```

### `Trajectory`
Container for atomic dynamics data.

```python
trajectory.positions   # (n_frames, n_atoms, 3)
trajectory.velocities  # (n_frames, n_atoms, 3)
trajectory.atom_types  # Atomic numbers
trajectory.box_matrix  # (3, 3) simulation cell
trajectory.timestep    # Frame spacing in ps
```

### `MultisliceCalculator`
Compute exit wavefunctions via multislice algorithm.

```python
from pyslice.multislice.calculators import MultisliceCalculator

calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=0,           # mrad (0 = parallel beam)
    voltage_eV=100e3,     # Accelerating voltage
    sampling=0.1,         # Å/pixel
    slice_thickness=0.5,  # Å
    probe_positions=None, # Optional (N,2) array for STEM
)
wf_data = calc.run()
```

### `TACAWData`
Frequency-domain phonon analysis.

```python
from pyslice.postprocessing.tacaw_data import TACAWData

tacaw = TACAWData(wf_data)

# Analysis methods
tacaw.frequencies                        # Available frequencies (THz)
tacaw.spectral_diffraction(freq_THz)     # k-space intensity at frequency
tacaw.dispersion(kx_path, ky_path)       # Phonon dispersion along k-path
tacaw.spectrum_image(freq_THz)           # Real-space map at frequency (STEM)
```

### `HAADFData`
STEM imaging analysis.

```python
from pyslice.postprocessing.haadf_data import HAADFData

haadf = HAADFData(wf_data)
haadf.calculateADF(inner_mrad=60, outer_mrad=200)
haadf.plot()
```

## Examples

See the `tests/` directory for detailed examples:
- `00_probe.py` - Probe wavefunction visualization
- `01_potentials.py` - Atomic potential calculations
- `04_haadf.py` - HAADF-STEM imaging
- `05_tacaw.py` - TACAW phonon spectroscopy
- `06_loaders.py` - Loading various file formats
- `15_molecular_dynamics.py` - MD with ORB potentials

## Requirements

**Core:**
- Python 3.10+
- NumPy, SciPy, Matplotlib
- ASE (Atomic Simulation Environment)
- OVITO

**Recommended:**
- PyTorch (GPU acceleration)

## License

MIT License - see LICENSE file for details.
