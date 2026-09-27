import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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
    max_generated_candidates: int = 1000,
    placement_mode: Literal[
        "heuristic", "random", "random_site_heuristic_placement"
    ] = "heuristic",
    random_seed: int = 0,
    interstitial_gap: float = 0.1,
    surface_layer_tolerance: float = 0.5,
    *,
    include_candidate_details: bool = True,
) -> Dict[str, Any]:
    """Generate starting configurations, optionally omitting persisted details.

    Internal workflows retain the full candidate list by default. MCP callers
    use a compact result and hand its manifest_path to the relaxation tool.
    """
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
                str(max_generated_candidates),
                placement_mode,
                str(random_seed),
                str(interstitial_gap),
                str(surface_layer_tolerance),
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
    if result.get("success") is True and not include_candidate_details:
        result = {key: value for key, value in result.items() if key != "candidates"}
        result["message"] = (
            "Candidate counts are upper bounds. Fewer available sites, including "
            "after symmetry reduction in heuristic mode, are a successful outcome. "
            "Report the actual count and continue using the candidates in manifest_path; "
            "do not retry or change placement mode solely to reach the upper bound. "
            "Full candidate details are stored in manifest_path. Pass that path as "
            "candidate_manifest_path to relax_adsorption_candidates."
        )
    return result


def relax_adsorption_candidates(
    candidate_structure_paths: Optional[List[str]] = None,
    candidate_set: Optional[Dict[str, Any]] = None,
    candidate_manifest_path: Optional[str] = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    calculator_backend: Literal["fairchem"] = "fairchem",
    model: str = "uma-s-1p2",
    task: str = "oc20",
    device: Literal["auto", "cpu", "cuda"] = "auto",
    precision: Literal["float32", "float64"] = "float32",
    optimizer: Literal["LBFGS", "BFGS", "FIRE"] = "LBFGS",
    desorption_distance: float = 4.0,
    penetration_depth: float = 1.0,
    severe_slab_displacement: float = 2.0,
) -> Dict[str, Any]:
    """Relax and rank an equal-composition set of adsorption candidates."""
    request = {
        "candidate_structure_paths": candidate_structure_paths,
        "candidate_set": candidate_set,
        "candidate_manifest_path": candidate_manifest_path,
        "fmax": fmax,
        "max_steps": max_steps,
        "calculator_backend": calculator_backend,
        "model": model,
        "task": task,
        "device": device,
        "precision": precision,
        "optimizer": optimizer,
        "desorption_distance_A": desorption_distance,
        "penetration_depth_A": penetration_depth,
        "severe_slab_displacement_A": severe_slab_displacement,
    }
    try:
        process = subprocess.run(
            [
                UMA_PYTHON,
                ADSORPTION_RELAXATION_RUNNER,
                json.dumps(request),
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
