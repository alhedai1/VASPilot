import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .uma_calculate import UMA_PYTHON


ADSORPTION_CANDIDATE_RUNNER = str(
    Path(__file__).resolve().parents[2] / "scripts" / "adsorption_candidate_runner.py"
)
ADSORPTION_RELAXATION_RUNNER = str(
    Path(__file__).resolve().parents[2] / "scripts" / "adsorption_relaxation_runner.py"
)


def generate_adsorption_candidates(
    slab_structure_path: str,
    adsorbate_structure_path: str,
    binding_atom_indices: List[int],
    num_sites: int = 10,
    num_orientations_per_site: int = 1,
) -> Dict[str, Any]:
    """Generate adsorbate-on-slab starting configurations in the UMA environment."""
    try:
        process = subprocess.run(
            [
                UMA_PYTHON,
                ADSORPTION_CANDIDATE_RUNNER,
                slab_structure_path,
                adsorbate_structure_path,
                ",".join(str(index) for index in binding_atom_indices),
                str(num_sites),
                str(num_orientations_per_site),
            ],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "success": False,
            "error": f"Failed to start adsorption candidate subprocess: {exc}",
        }

    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        error = f"Adsorption candidate runner returned invalid JSON: {exc}"
        if process.stderr.strip():
            error += f"; stderr: {process.stderr.strip()}"
        return {"success": False, "error": error}

    if not isinstance(result, dict):
        return {
            "success": False,
            "error": "Adsorption candidate runner JSON result is not an object",
        }
    if process.returncode != 0 and result.get("success") is not False:
        return {
            "success": False,
            "error": process.stderr.strip()
            or f"Adsorption candidate subprocess exited with code {process.returncode}",
        }
    return result


def relax_adsorption_candidates(
    candidate_structure_paths: List[str],
    fmax: float = 0.05,
    max_steps: int = 200,
) -> Dict[str, Any]:
    """Relax and rank an equal-composition set of adsorption candidates."""
    try:
        process = subprocess.run(
            [
                UMA_PYTHON,
                ADSORPTION_RELAXATION_RUNNER,
                json.dumps(candidate_structure_paths),
                str(fmax),
                str(max_steps),
            ],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "success": False,
            "error": f"Failed to start adsorption relaxation subprocess: {exc}",
        }

    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        error = f"Adsorption relaxation runner returned invalid JSON: {exc}"
        if process.stderr.strip():
            error += f"; stderr: {process.stderr.strip()}"
        return {"success": False, "error": error}

    if not isinstance(result, dict):
        return {
            "success": False,
            "error": "Adsorption relaxation runner JSON result is not an object",
        }
    if process.returncode != 0 and result.get("success") is not False:
        return {
            "success": False,
            "error": process.stderr.strip()
            or f"Adsorption relaxation subprocess exited with code {process.returncode}",
        }
    return result
