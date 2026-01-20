"""Microscope-specific calculator utilities."""

from __future__ import annotations


def q_to_theta(q: float, wave_length: float = 0.0197) -> float:
	"""Convert scattering vector magnitude to scattering angle.

	Parameters
	----------
	q : float
		Scattering vector magnitude in inverse Angstroms.
	wave_length : float, optional
		Electron wavelength (Angstroms), by default 0.0197 (approx 300 keV).

	Returns
	-------
	float
		Scattering angle in radians.
	"""
	return wave_length*q/2
