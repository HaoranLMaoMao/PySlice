"""
Core trajectory data structure for molecular dynamics data.
trajectory.py has no backend dependency — it is pure NumPy data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms


@dataclass
class Trajectory:
    atom_types: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    box_matrix: np.ndarray
    timestep: float  # Timestep in picoseconds

    def __post_init__(self):
        self._validate_shapes()

    def _validate_shapes(self):
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError(f"positions must be (frames, atoms, 3), got {self.positions.shape}")
        if self.velocities.ndim != 3 or self.velocities.shape[2] != 3:
            raise ValueError(f"velocities must be (frames, atoms, 3), got {self.velocities.shape}")
        if self.atom_types.ndim != 1:
            raise ValueError(f"atom_types must be 1D, got {self.atom_types.ndim}D")
        if self.box_matrix.shape != (3, 3):
            raise ValueError(f"box_matrix must be (3, 3), got {self.box_matrix.shape}")

        n_frames_pos, n_atoms_pos = self.positions.shape[:2]
        n_frames_vel, n_atoms_vel = self.velocities.shape[:2]
        n_atoms_types = len(self.atom_types)

        if n_frames_pos != n_frames_vel:
            raise ValueError(f"Frame count mismatch: {n_frames_pos} vs {n_frames_vel}")
        if not (n_atoms_pos == n_atoms_vel == n_atoms_types):
            raise ValueError(f"Atom count mismatch: {n_atoms_pos}, {n_atoms_vel}, {n_atoms_types}")

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_atoms(self) -> int:
        return len(self.atom_types)

    @property
    def box_tilts(self) -> np.ndarray:
        return np.array([self.box_matrix[0, 1], self.box_matrix[0, 2], self.box_matrix[1, 2]])

    @property
    def extent(self) -> np.ndarray:
        """Cartesian bounding box (x, y, z) of the simulation cell in Angstroms."""
        a, b, c = self.box_matrix[0], self.box_matrix[1], self.box_matrix[2]
        corners = np.array([[0,0,0], a, b, c, a+b, a+c, b+c, a+b+c])
        return corners.max(axis=0) - corners.min(axis=0)

    def get_mean_positions(self) -> np.ndarray:
        """Mean position for each atom over all frames."""
        if self.n_frames == 0:
            return np.empty((0, 3), dtype=self.positions.dtype)
        return np.mean(self.positions, axis=0)

    def get_distplacements(self) -> np.ndarray:
        """Per-frame displacement from mean position."""
        return self.positions - self.get_mean_positions()[None, :, :]

    def slice_timesteps(self, i1: int = 0, i2: Optional[int] = None, ith: int = 1) -> Trajectory:
        """Slice trajectory to a range of frames.

        Args:
            i1:  first frame index to keep (or a list/tuple of specific indices)
            i2:  first frame index to exclude (default: include last frame)
            ith: stride — 2 = every other frame, 3 = every third, etc.
        """
        if isinstance(i1, (list, tuple, np.ndarray)):
            return self.select_timesteps(i1)
        if i2 is None:
            i2 = len(self.positions)
        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[i1:i2:ith],
            velocities=self.velocities[i1:i2:ith],
            box_matrix=self.box_matrix,
            timestep=self.timestep * ith,
        )

    def select_timesteps(self, indices) -> Trajectory:
        """Select specific frames by index array."""
        if indices is None:
            indices = slice(0, len(self.positions))
        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[indices],
            velocities=self.velocities[indices],
            box_matrix=self.box_matrix,
            timestep=0.0,
        )

    def get_random_timesteps(self, N: int = 1, seed: Optional[int] = None) -> Trajectory:
        """Return N randomly selected frames (useful for frozen-phonon sampling).

        Args:
            N:    number of frames to keep
            seed: numpy random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(len(self.positions))
        np.random.shuffle(indices)
        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[indices[:N]],
            velocities=self.velocities[indices[:N]],
            box_matrix=self.box_matrix,
            timestep=self.timestep,
        )

    def tile_positions(self, repeats: Tuple[int, int, int],
                       trajectories: Optional[list] = None) -> Trajectory:
        """Tile the simulation cell by repeating in 3D space."""
        nx, ny, nz = repeats
        offsets = [
            np.array([i, j, k]) @ self.box_matrix
            for i in range(nx) for j in range(ny) for k in range(nz)
        ]
        if trajectories is None or len(trajectories) != len(offsets):
            trajectories = [self] * len(offsets)

        tiled_positions  = [t.positions + o for t, o in zip(trajectories, offsets)]
        tiled_velocities = [t.velocities    for t     in trajectories]
        tiled_types      = [t.atom_types    for t     in trajectories]

        new_box = self.box_matrix.copy()
        new_box[0, :] *= nx
        new_box[1, :] *= ny
        new_box[2, :] *= nz

        return Trajectory(
            atom_types=np.concatenate(tiled_types),
            positions=np.concatenate(tiled_positions, axis=1),
            velocities=np.concatenate(tiled_velocities, axis=1),
            box_matrix=new_box,
            timestep=self.timestep,
        )

    def swap_axes(self, axes) -> Trajectory:
        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[:, :, axes],
            velocities=self.velocities[:, :, axes],
            box_matrix=self.box_matrix[axes, :][:, axes],
            timestep=self.timestep,
        )

    def tilt_positions(self, alpha: float = 0, beta: float = 0) -> Trajectory:
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta),  np.sin(beta)
        Ra = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        Rb = np.array([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
        nt, na, _ = self.positions.shape
        R = Rb @ Ra
        pos = (self.positions.reshape(nt * na, 3) @ R.T).reshape(nt, na, 3)
        vel = (self.velocities.reshape(nt * na, 3) @ R.T).reshape(nt, na, 3)
        return Trajectory(
            atom_types=self.atom_types,
            positions=pos,
            velocities=vel,
            box_matrix=self.box_matrix,
            timestep=self.timestep,
        )

    def slice_positions(self,
                        x_range: Optional[Tuple[float, float]] = None,
                        y_range: Optional[Tuple[float, float]] = None,
                        z_range: Optional[Tuple[float, float]] = None) -> Trajectory:
        """Keep only atoms whose mean position falls within specified ranges."""
        if self.n_atoms == 0:
            return self
        x_range = self._validate_range(x_range, "X")
        y_range = self._validate_range(y_range, "Y")
        z_range = self._validate_range(z_range, "Z")
        if all(r is None for r in [x_range, y_range, z_range]):
            return self

        mean_pos = self.get_mean_positions()
        mask = np.ones(self.n_atoms, dtype=bool)
        for axis, rng in enumerate([x_range, y_range, z_range]):
            if rng is not None:
                mask &= (mean_pos[:, axis] >= rng[0]) & (mean_pos[:, axis] <= rng[1])

        return Trajectory(
            atom_types=self.atom_types[mask],
            positions=self.positions[:, mask, :],
            velocities=self.velocities[:, mask, :],
            box_matrix=self.box_matrix,
            timestep=self.timestep,
        )

    def _validate_range(self, range_val, axis_name):
        if range_val is None:
            return None
        lo, hi = range_val
        if lo > hi:
            raise ValueError(f"{axis_name} range invalid: min={lo} > max={hi}")
        return range_val

    def random_frames(self, N: int, seed: Optional[int] = None) -> Trajectory:
        """Return a new Trajectory with N randomly selected frames."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.n_frames, size=N, replace=False)
        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[indices],
            velocities=self.velocities[indices],
            box_matrix=self.box_matrix,
            timestep=self.timestep,
        )

    def generate_random_displacements(self, n_displacements: int, sigma: float,
                                      seed: Optional[int] = None) -> Trajectory:
        rng = np.random.default_rng(seed)
        na = self.n_atoms
        dxyz = rng.normal(0, sigma / np.sqrt(3), size=(n_displacements, na, 3))
        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[0] + dxyz,
            velocities=np.broadcast_to(self.velocities[0], (n_displacements, na, 3)).copy(),
            box_matrix=self.box_matrix,
            timestep=self.timestep,
        )

    def rotate_to(self, direction: Tuple[int, int, int]) -> Trajectory:
        """Rotate so that the given crystallographic direction aligns with z."""
        h, k, l = direction
        direction_cart = self.box_matrix.T @ np.array([h, k, l], dtype=float)
        direction_cart /= np.linalg.norm(direction_cart)
        z_axis = np.array([0.0, 0.0, 1.0])
        dot = np.dot(direction_cart, z_axis)

        if np.abs(dot - 1.0) < 1e-10:
            rotation_matrix = np.eye(3)
        elif np.abs(dot + 1.0) < 1e-10:
            rotation_matrix = np.diag([1.0, -1.0, -1.0])
        else:
            axis = np.cross(direction_cart, z_axis)
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            kx, ky, kz = axis
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        nt, na, _ = self.positions.shape
        pos = (self.positions.reshape(-1, 3) @ rotation_matrix.T).reshape(nt, na, 3)
        vel = (self.velocities.reshape(-1, 3) @ rotation_matrix.T).reshape(nt, na, 3)
        return Trajectory(
            atom_types=self.atom_types,
            positions=pos,
            velocities=vel,
            box_matrix=rotation_matrix @ self.box_matrix,
            timestep=self.timestep,
        )

    def to_ase(self) -> Atoms:
        return Atoms(
            ''.join(self.atom_types),
            positions=self.positions[0],
            cell=np.diag(self.box_matrix),
            pbc=True,
        )

    def plot(self, timestep: int = 0, view: str = '3d',
             alpha: float = 0.6, size: int = 20) -> None:
        import matplotlib.pyplot as plt
        COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        unique_types = sorted(set(self.atom_types))

        if view == '3d':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')
            for at, c in zip(unique_types, COLORS):
                mask = self.atom_types == at
                ax.scatter(*self.positions[timestep, mask].T,
                           c=c, s=size, alpha=alpha, label=str(at), edgecolors='none')
            ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)'); ax.set_zlabel('Z (Å)')
            ax.legend()
            box = self.box_matrix
            corners = np.array([
                [0,0,0],[box[0,0],0,0],[box[0,0],box[1,1],0],[0,box[1,1],0],
                [0,0,box[2,2]],[box[0,0],0,box[2,2]],[box[0,0],box[1,1],box[2,2]],[0,box[1,1],box[2,2]]
            ])
            for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                ax.plot3D(*corners[[i,j]].T, 'k-', alpha=0.3, linewidth=1)
        else:
            axis_map = {'xy': (0, 1, 'X (Å)', 'Y (Å)'),
                        'xz': (0, 2, 'X (Å)', 'Z (Å)'),
                        'yz': (1, 2, 'Y (Å)', 'Z (Å)')}
            if view not in axis_map:
                raise ValueError(f"view must be '3d', 'xy', 'xz', or 'yz', got {view!r}")
            i1, i2, xl, yl = axis_map[view]
            fig, ax = plt.subplots(figsize=(8, 8))
            for at, c in zip(unique_types, COLORS):
                mask = self.atom_types == at
                ax.scatter(self.positions[timestep, mask, i1],
                           self.positions[timestep, mask, i2],
                           c=c, s=size, alpha=alpha, label=str(at), edgecolors='none')
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.legend(); ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()
