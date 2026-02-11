"""
HAADF-STEM & CBED: Frozen Phonon vs Real MD Trajectory
=======================================================

Compare two approaches to multislice simulation:

  Part A — Static structure: load a CIF, tile, add Gaussian frozen-phonon
           displacements → CBED + HAADF
  Part B — Real dynamics: load a LAMMPS MD trajectory with true thermal
           motion → CBED + HAADF
  Part C — Aberrations: re-run Part A with spherical aberration and
           2-fold astigmatism to see their effect on the HAADF image

Same scan grid and microscope parameters in both cases so the results
are directly comparable.

Input files:
    ../tests/inputs/hBN_monolayer.cif
    ../tests/inputs/hBN_truncated.lammpstrj

Requirements:
    pip install pyslice
"""

import os
import numpy as np
from pyslice import Loader, MultisliceCalculator, HAADFData

os.makedirs("outputs", exist_ok=True)

# hBN lattice parameters (Å)
a = 2.491   # hexagonal lattice parameter
b = 2.157   # inter-row spacing along y (= a√3/2)

# Shared scan grid: 48×48 probes over a 3×3 unit-cell window
probe_xs = np.linspace(a, 4 * a, 48)
probe_ys = np.linspace(b, 4 * b, 48)

# ============================= PART A ======================================
# Frozen phonon from CIF — static structure with random displacements
# ===========================================================================
print("=" * 60)
print("PART A: CIF + frozen phonon")
print("=" * 60)

trajectory_cif = Loader("../tests/inputs/hBN_monolayer.cif").load()
trajectory_cif = trajectory_cif.tile_positions([5, 5, 1])
trajectory_cif = trajectory_cif.generate_random_displacements(
    n_displacements=20, sigma=0.1, seed=0,
)
print(f"CIF supercell: {trajectory_cif.n_atoms} atoms, {trajectory_cif.n_frames} frozen-phonon frames")

calc = MultisliceCalculator()
calc.setup(
    trajectory_cif,
    aperture=30, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5,
    probe_xs=probe_xs, probe_ys=probe_ys,
    use_memmap=True,
    loop_probes=500,  # process probes in chunks of 500 to limit peak memory
)

wf_cif = calc.run()

wf_cif.plot_reciprocal(
    "outputs/cbed_frozen_phonon.png",
    whichProbe=0, powerscaling=0.1, extent=[-2, 2, -2, 2],
)
print("Saved CBED (frozen phonon)")

haadf_cif = HAADFData(wf_cif)
# inner=60 mrad, outer=200 mrad: collect high-angle scattered electrons
haadf_cif.calculateADF(inner_mrad=60, outer_mrad=200)
haadf_cif.plot("outputs/haadf_frozen_phonon.png")
print("Saved HAADF (frozen phonon)")

# ============================= PART B ======================================
# Real MD trajectory — true correlated thermal motion
# ===========================================================================
print()
print("=" * 60)
print("PART B: LAMMPS MD trajectory")
print("=" * 60)

trajectory_md = Loader(
    "../tests/inputs/hBN_truncated.lammpstrj",
    timestep=0.005,
    atom_mapping={1: "B", 2: "N"},
).load()

trajectory_md = trajectory_md.slice_positions([0, 10 * a], [0, 10 * b])
trajectory_md = trajectory_md.get_random_timesteps(20, seed=5)
print(f"MD trajectory: {trajectory_md.n_atoms} atoms, {trajectory_md.n_frames} frames")

calc = MultisliceCalculator()
calc.setup(
    trajectory_md,
    aperture=30, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5,
    probe_xs=probe_xs, probe_ys=probe_ys,
    use_memmap=True,
    loop_probes=500,  # process probes in chunks of 500 to limit peak memory
)

wf_md = calc.run()

wf_md.plot_reciprocal(
    "outputs/cbed_md_trajectory.png",
    whichProbe=0, powerscaling=0.1, extent=[-2, 2, -2, 2],
)
print("Saved CBED (MD trajectory)")

haadf_md = HAADFData(wf_md)
haadf_md.calculateADF(inner_mrad=60, outer_mrad=200)
haadf_md.plot("outputs/haadf_md_trajectory.png")
print("Saved HAADF (MD trajectory)")

# ============================= PART C ======================================
# Aberrations — same frozen-phonon structure as Part A, with lens aberrations
# ===========================================================================
# Aberrations are applied to the probe after setup using calc.base_probe.aberrate().
# The argument is a dict of Cnm coefficients (following abTEM convention):
#
#   Rotationally symmetric:   "C30": value           (e.g. spherical aberration in Å)
#   With orientation angle:   "C12": (value, angle)  (e.g. 2-fold astigmatism)
#
# Common coefficients:
#   C10 — defocus                   C12 — 2-fold astigmatism
#   C21 — axial coma                C23 — 3-fold astigmatism
#   C30 — 3rd-order spherical       C32 — axial star
#   C34 — 4-fold astigmatism
print()
print("=" * 60)
print("PART C: Frozen phonon + aberrations")
print("=" * 60)

calc = MultisliceCalculator()
calc.setup(
    trajectory_cif,
    aperture=30, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5,
    probe_xs=probe_xs, probe_ys=probe_ys,
    use_memmap=True,
    loop_probes=500,
)

# Apply aberrations to the probe before running
calc.base_probe.aberrate({
    "C30": 1e3,              # 1000 Å spherical aberration
    "C12": (1e2, 0.0),       # 100 Å 2-fold astigmatism at 0°
})

wf_aberr = calc.run()

wf_aberr.plot_reciprocal(
    "outputs/cbed_aberrated.png",
    whichProbe=0, powerscaling=0.1, extent=[-2, 2, -2, 2],
)
print("Saved CBED (aberrated)")

haadf_aberr = HAADFData(wf_aberr)
haadf_aberr.calculateADF(inner_mrad=60, outer_mrad=200)
haadf_aberr.plot("outputs/haadf_aberrated.png")
print("Saved HAADF (aberrated)")
