#!/usr/bin/env python3
"""
Real-Space Phonon Mapping

Demonstrates phonon localization in real space:
1. Si-Ge interface: phonons localized by material region
2. Si-doped graphene: localized defect modes
"""

import os
from pathlib import Path

import gc

import matplotlib.pyplot as plt
import numpy as np

from ase import Atoms
from ase.build import bulk, surface

from pyslice import Loader, ORBMDCalculator, MultisliceCalculator, TACAWData

# =============================================================================
# CONFIGURATION
# =============================================================================

SYSTEMS = {
    "Si-Ge": {
        "frequencies": [15.0, 9.0],
        "labels": ["Si modes (15 THz)", "Ge modes (9 THz)"],
        "save_interval": 4,
    },
    "Si-Graphene": {
        "frequencies": [45.0, 12.0],
        "labels": ["C modes (45 THz)", "Si localized (12 THz)"],
        "save_interval": 2,
    },
}

SI_GE_LAYERS = 6
SI_GE_XY = 8
GRAPHENE_SIZE = 20

PROBE_GRID = 8  # reduced for efficiency; 32x32 was used for publication figures
APERTURE = 30.0  # mrad
SCAN_HALF_WIDTH = 4.0  # Å

TEMPERATURE = 300
MD_TIMESTEP = 2.0  # fs
PRODUCTION_STEPS = 2000

VOLTAGE_EV = 100e3
SAMPLING = 0.1
SLICE_THICKNESS = 0.5
PROBE_BATCH_SIZE = 4

SKIP_MD = True
SKIP_SYSTEMS = {"Si-Graphene": False, "Si-Ge": False}  # Set True to skip
CACHED_WEIGHTS_PATH = os.environ.get("ORB_WEIGHTS_PATH")

# =============================================================================
# STRUCTURE BUILDERS
# =============================================================================


def build_si_ge_interface():
    """Build Si-Ge heterostructure along [110] zone axis."""
    a_si = 5.431
    si_cubic = bulk('Si', 'diamond', a=a_si, cubic=True)
    si_110 = surface(si_cubic, (1, 1, 0), layers=2, vacuum=0, periodic=True)
    crystal = si_110 * (SI_GE_LAYERS * 2, SI_GE_XY, SI_GE_XY)
    crystal.wrap()

    interface_x = crystal.cell[0, 0] / 2
    symbols = ['Si' if pos[0] < interface_x else 'Ge' for pos in crystal.positions]
    crystal.set_chemical_symbols(symbols)
    crystal.info['interface_x'] = interface_x
    return crystal


def build_si_graphene():
    """Build graphene with single Si substitution at center."""
    d = 1.42
    vacuum = 15.0
    z_mid = vacuum / 2

    cell_x, cell_y = 3 * d, np.sqrt(3) * d
    positions = [[0, 0, z_mid], [d/2, cell_y/2, z_mid],
                 [3*d/2, cell_y/2, z_mid], [2*d, 0, z_mid]]

    gr = Atoms('C4', positions=positions, cell=[cell_x, cell_y, vacuum], pbc=True)
    supercell = gr * (GRAPHENE_SIZE, GRAPHENE_SIZE, 1)

    center = np.array([supercell.cell[0, 0], supercell.cell[1, 1]]) / 2
    distances = np.linalg.norm(supercell.positions[:, :2] - center, axis=1)
    center_idx = np.argmin(distances)

    symbols = supercell.get_chemical_symbols()
    symbols[center_idx] = 'Si'
    supercell.set_chemical_symbols(symbols)
    supercell.info['defect_position'] = supercell.positions[center_idx].copy()
    return supercell


# =============================================================================
# WORKFLOW
# =============================================================================


def run_md(atoms, output_dir: Path, save_interval: int):
    """Run MD with structure relaxation."""
    traj_file = output_dir / "production.traj"
    timestep_ps = MD_TIMESTEP * save_interval / 1000

    if SKIP_MD and traj_file.exists():
        return Loader(filename=str(traj_file), timestep=timestep_ps).load()

    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    from ase.optimize import FIRE

    orb = pretrained.orb_v3_conservative_inf_omat(weights_path=CACHED_WEIGHTS_PATH)
    atoms.calc = ORBCalculator(orb)
    FIRE(atoms, logfile=str(output_dir / "relax.log")).run(fmax=0.05, steps=500)

    md = ORBMDCalculator(
        model_name="orb-v3-conservative-inf-omat",
        weights_path=CACHED_WEIGHTS_PATH,
    )
    md.setup(
        atoms=atoms,
        temperature=TEMPERATURE,
        timestep=MD_TIMESTEP,
        ensemble="nvt",
        friction=0.2,
        production_ensemble="nvt",
        production_friction=0.001,
        production_relaxation_steps=100,
        production_steps=PRODUCTION_STEPS,
        save_interval=save_interval,
        output_dir=str(output_dir),
    )
    return md.run()


def create_probe_grid(atoms):
    """Create probe grid centered on interface/defect."""
    cell = atoms.cell.lengths()
    if 'defect_position' in atoms.info:
        cx, cy = atoms.info['defect_position'][:2]
    else:
        cx, cy = atoms.info['interface_x'], cell[1] / 2

    x_min = max(0.5, cx - SCAN_HALF_WIDTH)
    x_max = min(cell[0] - 0.5, cx + SCAN_HALF_WIDTH)
    y_min = max(0.5, cy - SCAN_HALF_WIDTH)
    y_max = min(cell[1] - 0.5, cy + SCAN_HALF_WIDTH)

    xs = np.linspace(x_min, x_max, PROBE_GRID)
    ys = np.linspace(y_min, y_max, PROBE_GRID)
    return [(x, y) for y in ys for x in xs], [x_min, x_max, y_min, y_max]


def compute_spectrum_images(trajectory, probe_positions, frequencies, output_dir):
    """Compute real-space maps at specified frequencies.

    With many probes, the full wavefunction array (n_probes x n_frames x nx x ny)
    can exceed GPU memory. Two approaches:

    1. use_memmap=True + loop_probes: The wavefunction array is memory-mapped to
       local disk and probes are processed in chunks. This is the cleanest approach
       but requires a filesystem that supports mmap (local SSD, NVMe, etc.).
       Network filesystems like Lustre/GPFS/CFS will fail with ENOSYS.

    2. Manual batching (used here): Process probe batches in separate calculator
       calls, keeping only one small batch in memory at a time. Works on any
       filesystem.
    """
    # -- Approach 1: use_memmap + loop_probes (for local filesystems) -----------
    # calc = MultisliceCalculator()
    # calc.setup(
    #     trajectory,
    #     aperture=APERTURE,
    #     voltage_eV=VOLTAGE_EV,
    #     sampling=SAMPLING,
    #     slice_thickness=SLICE_THICKNESS,
    #     probe_positions=probe_positions,
    #     save_path=output_dir,
    #     cleanup_temp_files=True,
    #     use_memmap=True,
    #     loop_probes=PROBE_BATCH_SIZE,
    # )
    # tacaw = TACAWData(calc.run(), chunkFFT=True)
    # return {f: tacaw.spectrum_image(f) for f in frequencies}

    # -- Approach 2: Manual batching (for network filesystems like CFS) ---------
    n_probes = len(probe_positions)
    n_batches = (n_probes + PROBE_BATCH_SIZE - 1) // PROBE_BATCH_SIZE
    maps = {f: np.zeros(n_probes) for f in frequencies}

    for batch_idx in range(n_batches):
        start = batch_idx * PROBE_BATCH_SIZE
        end = min(start + PROBE_BATCH_SIZE, n_probes)

        calc = MultisliceCalculator()
        calc.setup(
            trajectory,
            aperture=APERTURE,
            voltage_eV=VOLTAGE_EV,
            sampling=SAMPLING,
            slice_thickness=SLICE_THICKNESS,
            probe_positions=probe_positions[start:end],
            save_path=output_dir,
            cleanup_temp_files=True,
        )
        tacaw = TACAWData(calc.run(), chunkFFT=True)

        for freq in frequencies:
            maps[freq][start:end] = tacaw.spectrum_image(freq)

    return maps


def plot_spectrum_image(intensity, freq, label, extents, name, output_dir,
                        defect_pos=None, interface_x=None):
    """Plot a single spectrum image."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(intensity, cmap="inferno", origin="lower", aspect="equal",
                   extent=extents, vmin=0, vmax=1)

    if defect_pos is not None:
        ax.plot(defect_pos[0], defect_pos[1], 'w+', ms=15, mew=2)

    if interface_x is not None:
        ax.axvline(interface_x, color="white", ls="--", lw=2, alpha=0.8)

    ax.set_xlabel(r"x ($\AA$)")
    ax.set_ylabel(r"y ($\AA$)")
    ax.set_title(f"{name} - {label}", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Normalized Intensity", shrink=0.8)
    plt.savefig(output_dir / f"{name}_{freq:.0f}THz.png", dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    output_base = Path(__file__).parent / "outputs" / "real_space_phonons"
    output_base.mkdir(parents=True, exist_ok=True)

    for name, params in SYSTEMS.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")

        if SKIP_SYSTEMS.get(name, False):
            print(f"Skipping {name}...")
            continue

        output_dir = output_base / name
        output_dir.mkdir(exist_ok=True)

        if name == "Si-Graphene":
            atoms = build_si_graphene()
        else:
            atoms = build_si_ge_interface()

        trajectory = run_md(atoms, output_dir, params["save_interval"])
        probes, extents = create_probe_grid(atoms)
        maps = compute_spectrum_images(trajectory, probes, params["frequencies"], output_dir)

        defect_pos = atoms.info.get('defect_position', [None, None])[:2]
        interface_x = atoms.info.get('interface_x')

        # Save results
        for freq in params["frequencies"]:
            np.save(output_dir / f"spectrum_image_{freq:.1f}THz.npy", maps[freq])
        np.save(output_dir / "probe_positions.npy", np.array(probes))
        np.save(output_dir / "grid_metadata.npy", {
            'grid_shape': (PROBE_GRID, PROBE_GRID),
            'extents': extents,
            'defect_pos': defect_pos if defect_pos[0] else None,
            'interface_x': interface_x,
        })

        # Plot
        for freq, label in zip(params["frequencies"], params["labels"]):
            intensity = maps[freq].reshape((PROBE_GRID, PROBE_GRID))
            intensity /= intensity.max()
            plot_spectrum_image(intensity, freq, label, extents, name, output_dir,
                                defect_pos if defect_pos[0] else None, interface_x)

        # Clear GPU memory
        del trajectory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
