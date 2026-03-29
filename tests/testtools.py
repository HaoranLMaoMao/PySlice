"""
Test utilities for comparing simulation outputs across runs.
"""
from __future__ import annotations

import os
import numpy as np

from pyslice.backend import to_numpy


# to have matplotlib withoput display issues on headless machines
os.environ['MPLBACKEND'] = 'Agg'


def differ_phase(ary, filename: str, label: str, tol: float = 1e-6) -> None:
    """Compare the phase of an array against a saved reference."""
    ary = to_numpy(ary)
    if not os.path.exists(filename):
        print(f"Reference file for {label!r} (phase) does not exist — creating it now.")
        np.save(filename, ary)
        return
    previous = np.load(filename)
    F = np.angle(ary)
    D = np.angle(previous)
    denom = np.sum(F ** 2)
    if denom == 0:
        print(f"WARNING: {label} (phase) — reference array phase is all zeros; skipping comparison.")
        return
    dz = np.sum((F - D) ** 2) / denom
    if dz > tol:
        print(f"ERROR! {label} (phase) does not match previous run — residual {dz * 100:.4f}%")
    else:
        print(f"OK: {label} (phase) matches previous run (residual {dz * 100:.6f}%)")


def differ_abs(ary, filename: str, label: str, tol: float = 1e-6) -> None:
    """Compare the absolute value of an array against a saved reference."""
    ary = to_numpy(ary)
    if not os.path.exists(filename):
        print(f"Reference file for {label!r} (abs) does not exist — creating it now.")
        np.save(filename, ary)
        return
    previous = np.load(filename)
    F = np.abs(ary)
    D = np.abs(previous)
    denom = np.sum(F ** 2)
    if denom == 0:
        print(f"WARNING: {label} (abs) — reference array is all zeros; skipping comparison.")
        return
    dz = np.sum((F - D) ** 2) / denom
    if dz > tol:
        print(f"ERROR! {label} (abs) does not match previous run — residual {dz * 100:.4f}%")
    else:
        print(f"OK: {label} (abs) matches previous run (residual {dz * 100:.6f}%)")


def differ(ary, filename: str, label: str, tol: float = 1e-6) -> None:
    """
    Compare an array against a saved reference, creating it if absent.

    The residual is scale- and near-zero-resistant:
        dz = Σ(|F| - |D|)² / Σ|F|²

    Args:
        ary:      Array to test (any backend type — converted to numpy internally).
        filename: Path to the reference .npy file.
        label:    Human-readable name printed in the error message.
        tol:      Residual threshold above which a mismatch is reported.
    """
    ary = to_numpy(ary)

    if not os.path.exists(filename):
        print(f"Reference file for {label!r} does not exist — creating it now.")
        np.save(filename, ary)
        return

    previous = np.load(filename)
    F = np.abs(ary)
    D = np.abs(previous)
    denom = np.sum(F ** 2)
    if denom == 0:
        print(f"WARNING: {label} — reference array is all zeros; skipping comparison.")
        return
    dz = np.sum((F - D) ** 2) / denom
    if dz > tol:
        print(f"ERROR! {label} does not match previous run — residual {dz * 100:.4f}%")
    else:
        print(f"OK: {label} matches previous run (residual {dz * 100:.6f}%)")
