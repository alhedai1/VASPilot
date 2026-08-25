from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
from ase.io import read


def _finite_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def load_trajectory_data(trajectory_path: str, max_frames: int = 2001) -> Dict[str, Any]:
    """Read an ASE-supported trajectory into JSON-safe visualization data."""
    path = Path(trajectory_path)
    if path.suffix.lower() not in {".traj", ".extxyz"}:
        raise ValueError("Trajectory must have a .traj or .extxyz extension")

    frames = read(path, index=":")
    if not frames:
        raise ValueError("Trajectory contains no frames")
    if len(frames) > max_frames:
        raise ValueError(f"Trajectory exceeds the {max_frames}-frame viewer limit")

    first = frames[0]
    symbols = first.get_chemical_symbols()
    if any(frame.get_chemical_symbols() != symbols for frame in frames):
        raise ValueError("Atom ordering changes between trajectory frames")

    fixed_indices = set()
    for constraint in first.constraints:
        get_indices = getattr(constraint, "get_indices", None)
        if callable(get_indices):
            fixed_indices.update(int(index) for index in get_indices())

    tags = first.get_tags()
    adsorbate_indices = [index for index, tag in enumerate(tags) if int(tag) == 2]
    adsorbate_indices_inferred = False
    if not adsorbate_indices:
        counts = Counter(symbols)
        rare_symbols = {
            symbol
            for symbol, count in counts.items()
            if count <= max(2, len(symbols) // 20)
        }
        adsorbate_indices = [
            index for index, symbol in enumerate(symbols) if symbol in rare_symbols
        ]
        adsorbate_indices_inferred = True

    frame_data = []
    for frame_index, atoms in enumerate(frames):
        try:
            energy = _finite_float(atoms.get_potential_energy())
        except Exception:
            energy = None
        try:
            forces = np.asarray(atoms.get_forces(), dtype=float)
            max_force = _finite_float(np.linalg.norm(forces, axis=1).max())
        except Exception:
            max_force = None
        frame_data.append(
            {
                "index": frame_index,
                "positions": np.asarray(atoms.positions, dtype=float).tolist(),
                "energy_eV": energy,
                "max_force_eV_per_A": max_force,
            }
        )

    return {
        "trajectory_path": str(path.resolve()),
        "frame_count": len(frames),
        "atom_count": len(first),
        "symbols": symbols,
        "cell": np.asarray(first.cell, dtype=float).tolist(),
        "pbc": [bool(value) for value in first.pbc],
        "fixed_atom_indices": sorted(fixed_indices),
        "adsorbate_atom_indices": adsorbate_indices,
        "adsorbate_indices_inferred": adsorbate_indices_inferred,
        "frames": frame_data,
    }
