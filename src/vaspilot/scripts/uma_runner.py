"""Relax an atomic structure with the UMA model and report the result as JSON."""

import contextlib
import io
import json
import sys
from pathlib import Path
import sys

MODEL = "uma-s-1p2"
TASK = "omat"
FMAX = 0.05
MAX_STEPS = 200


def relax_structure(input_path: Path) -> dict:
    """Run a full atomic and cell relaxation for *input_path*."""
    from ase.filters import FrechetCellFilter
    from ase.io import read, write
    from ase.optimize import FIRE
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    import numpy as np

    if not input_path.is_file():
        raise FileNotFoundError(f"Input structure not found: {input_path}")

    output_path = input_path.with_name(f"{input_path.stem}_relaxed.cif")

    predictor = pretrained_mlip.get_predict_unit(
        MODEL,
        device="cuda",
        inference_settings=InferenceSettings(compile=False),
    )
    calculator = FAIRChemCalculator(predictor, task_name=TASK)

    atoms = read(input_path)
    atoms.calc = calculator
    initial_energy = float(atoms.get_potential_energy())

    optimizer = FIRE(FrechetCellFilter(atoms), logfile=None)
    optimizer.run(fmax=FMAX, steps=MAX_STEPS)

    final_energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0

    write(output_path, atoms, format="cif")

    return {
        "success": True,
        "model": MODEL,
        "task": TASK,
        "input_structure": str(input_path),
        "relaxed_structure": str(output_path),
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "steps": int(optimizer.nsteps),
        "max_atomic_force_eV_per_A": max_force,
        "stress_eV_per_A3": [float(component) for component in stress],
    }


def main() -> int:
    print(sys.executable, file=sys.stderr)
    result = None
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: python uma_runner.py <input_structure>")

        # Some dependencies print status messages; keep stdout reserved for JSON.
        with contextlib.redirect_stdout(io.StringIO()):
            result = relax_structure(Path(sys.argv[1]).expanduser())
        exit_code = 0
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
        exit_code = 1

    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
