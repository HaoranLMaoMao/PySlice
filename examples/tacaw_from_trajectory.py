"""
TACAW from Trajectory: Load MD → Multislice → Spectral Diffraction
===================================================================

Run the TACAW workflow starting from a pre-existing MD trajectory.

Unlike tacaw_pipeline.py (which runs MD from scratch), this example loads
a LAMMPS dump file directly and proceeds to multislice + TACAW analysis.

Input file:
    ../tests/inputs/hBN_truncated.lammpstrj

Requirements:
    pip install pyslice
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pyslice import Loader, MultisliceCalculator, TACAWData

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the MD trajectory
# ---------------------------------------------------------------------------
trajectory = Loader(
    "../tests/inputs/hBN_truncated.lammpstrj",
    timestep=0.005,  # ps between frames (5 fs)
    atom_mapping={1: "B", 2: "N"},
).load()

print(f"Trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")

# ---------------------------------------------------------------------------
# 2. Parallel-beam multislice over all frames
# ---------------------------------------------------------------------------
calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=0,            # Parallel beam (plane wave)
    voltage_eV=100e3,
    sampling=0.1,
    slice_thickness=0.5,
)

wf_data = calc.run()
print(f"Exit-wave shape: {wf_data.array.shape}")

# ---------------------------------------------------------------------------
# 3. TACAW: time → frequency domain
# ---------------------------------------------------------------------------
tacaw = TACAWData(wf_data)
freqs = tacaw.frequencies
df = freqs[1] - freqs[0]
print(f"Frequency range: 0 – {freqs[-1]:.1f} THz  "
      f"({len(freqs)} bins, Δf = {abs(df):.1f} THz)")

# Spectral diffraction at 15 THz, cropped to ±2 Å⁻¹
Z = tacaw.spectral_diffraction(15.0)
tacaw.plot(Z ** 0.1, "kx", "ky",
           extent=[-2, 2, -2, 2],
           filename="outputs/tacaw_hbn_15THz.png")
print("Saved spectral diffraction at 15 THz")

# ---------------------------------------------------------------------------
# 4. Phonon dispersion: Gamma → Gamma → Gamma
# ---------------------------------------------------------------------------
# The reciprocal lattice period along kx depends on the crystal orientation
# inside the simulation cell. For this hBN trajectory it is 2π/(3a).
a_hbn = 2.491  # hBN lattice parameter (Å)
G_x = 2 * np.pi / (3 * a_hbn)
kx_path = np.linspace(-G_x, G_x, 200)
ky_path = np.zeros_like(kx_path)
dispersion = tacaw.dispersion(kx_path, ky_path)

# Plot positive frequencies only
pos_mask = freqs >= 0
fig, ax = plt.subplots()
ax.imshow(
    np.abs(dispersion[pos_mask, :]) ** 0.125,
    cmap="inferno", aspect="auto", origin="lower",
    extent=[kx_path[0], kx_path[-1], freqs[pos_mask][0], freqs[pos_mask][-1]],
)
ax.set_xlabel("kx ($\\AA^{-1}$)")
ax.set_ylabel("frequency (THz)")
fig.savefig("outputs/tacaw_hbn_dispersion.png")
plt.close(fig)
print("Saved phonon dispersion Gamma → Gamma → Gamma")
