"""
Probe Aberrations: Effect on HAADF-STEM
=======================================

Show how lens aberrations affect HAADF imaging by comparing an ideal
probe against one with spherical aberration and 2-fold astigmatism.

Aberrations are applied after setup using calc.base_probe.aberrate().
The argument is a dict of Cnm coefficients (following abTEM convention):

  Rotationally symmetric:   "C30": value           (e.g. spherical aberration in Å)
  With orientation angle:   "C12": (value, angle)   (e.g. 2-fold astigmatism)

Common coefficients:
  C10 — defocus                   C12 — 2-fold astigmatism
  C21 — axial coma                C23 — 3-fold astigmatism
  C30 — 3rd-order spherical       C32 — axial star
  C34 — 4-fold astigmatism

Input file:
    ../tests/inputs/hBN_monolayer.cif
"""

import os
import numpy as np
from pyslice import Loader, MultisliceCalculator, HAADFData

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Build an hBN frozen-phonon supercell
# ---------------------------------------------------------------------------
trajectory = Loader("../tests/inputs/hBN_monolayer.cif").load()
trajectory = trajectory.tile_positions([10, 10, 1])
trajectory = trajectory.generate_random_displacements(
    n_displacements=20, sigma=0.1, seed=0,
)
print(f"Supercell: {trajectory.n_atoms} atoms, {trajectory.n_frames} frozen-phonon frames")

a = 2.491   # hBN lattice parameter (Å)
b = 2.157   # inter-row spacing (= a√3/2)
probe_xs = np.linspace(a, 4 * a, 48)
probe_ys = np.linspace(b, 4 * b, 48)

# ---------------------------------------------------------------------------
# 2. Ideal probe — no aberrations
# ---------------------------------------------------------------------------
print("=" * 60)
print("Ideal probe (no aberrations)")
print("=" * 60)

calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=30, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5,
    probe_xs=probe_xs, probe_ys=probe_ys,
    use_memmap=True,
    loop_probes=500,
)

wf_ideal = calc.run()

wf_ideal.plot_reciprocal(
    "outputs/cbed_ideal.png",
    whichProbe=0, powerscaling=0.1, extent=[-2, 2, -2, 2],
)

haadf_ideal = HAADFData(wf_ideal)
haadf_ideal.calculateADF(inner_mrad=60, outer_mrad=200)
haadf_ideal.plot("outputs/haadf_ideal.png")
print("Saved ideal CBED + HAADF")

# ---------------------------------------------------------------------------
# 3. Aberrated probe — spherical aberration + 2-fold astigmatism
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Aberrated probe (C30 + C12)")
print("=" * 60)

calc = MultisliceCalculator()
calc.setup(
    trajectory,
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

haadf_aberr = HAADFData(wf_aberr)
haadf_aberr.calculateADF(inner_mrad=60, outer_mrad=200)
haadf_aberr.plot("outputs/haadf_aberrated.png")
print("Saved aberrated CBED + HAADF")
