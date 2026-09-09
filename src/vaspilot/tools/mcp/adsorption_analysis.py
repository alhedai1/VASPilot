"""Select an UMA-ranked adsorption candidate for static VASP analysis."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1
SUPPORTED_ANALYSES = {"bader", "lobster"}
GEOMETRY_FAILURE_FLAGS = (
    "desorbed",
    "penetrated_surface",
    "fragmented",
    "severe_slab_reconstruction",
)
DEFAULT_INCAR = {
    "GGA": "PE",
    "ENCUT": 450,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "ISPIN": 1,
    "IBRION": -1,
    "NSW": 0,
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "ALGO": "Normal",
    "NELM": 160,
    "LREAL": False,
    "ISYM": -1,
    "LSORBIT": False,
    "LDIPOL": False,
    "LCHARG": True,
    "LAECHG": True,
    "LWAVE": True,
    "ADDGRID": True,
    "LORBIT": 11,
}


def _load_manifest(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Relaxation manifest not found: {path}")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("Relaxation manifest must contain an object")
    return result


def load_analysis_scf_submission(manifest_path: str) -> tuple[Path, Dict[str, Any]]:
    """Validate a prepared-analysis manifest and return its exact SCF arguments."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_manifest(path)
    if manifest.get("artifact_kind") != "adsorption_analysis_scf_preparation":
        raise ValueError("Expected an adsorption_analysis_scf_preparation artifact")
    if manifest.get("success") is not True:
        raise ValueError("SCF preparation manifest is not successful")
    if manifest.get("submitted") is not False:
        raise ValueError("SCF preparation manifest is already marked as submitted")

    arguments = manifest.get("vasp_scf_arguments")
    if not isinstance(arguments, dict):
        raise ValueError("SCF preparation manifest has no vasp_scf_arguments object")
    if set(arguments) != {"structure_path", "soc", "incar_tags", "kpoint_num"}:
        raise ValueError("vasp_scf_arguments has an unexpected schema")
    structure = Path(arguments["structure_path"]).expanduser().resolve()
    if not structure.is_file():
        raise FileNotFoundError(f"Prepared SCF structure not found: {structure}")
    expected_hash = (manifest.get("provenance") or {}).get("source_structure_sha256")
    if expected_hash and hashlib.sha256(structure.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Prepared SCF structure does not match its provenance hash")
    if arguments["soc"] is not False or not isinstance(arguments["incar_tags"], dict):
        raise ValueError("Prepared SCF arguments are invalid")
    mesh = arguments["kpoint_num"]
    if not isinstance(mesh, list) or len(mesh) != 3 or any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in mesh
    ):
        raise ValueError("Prepared SCF kpoint_num must contain three positive integers")

    prior = path.with_name("analysis_scf_submission.json")
    if prior.is_file() and json.loads(prior.read_text()).get("submitted") is True:
        raise ValueError("This SCF preparation artifact has already been submitted")
    return path, arguments


def write_analysis_scf_submission(
    manifest_path: Path, submission_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Persist the link between a preparation artifact and its VASP submission."""
    manifest = _load_manifest(manifest_path)
    success = submission_result.get("success") is True
    record = {
        "success": success,
        "artifact_kind": "adsorption_analysis_scf_submission",
        "preparation_artifact_id": manifest.get("artifact_id"),
        "preparation_manifest_path": str(manifest_path),
        "selected_candidate_id": manifest.get("selected_candidate_id"),
        "submitted": success,
        "calculation_id": submission_result.get("calculation_id"),
        "slurm_id": submission_result.get("slurm_id"),
        "calculation_directory": submission_result.get("calculate_path"),
        "status": submission_result.get("status"),
        "error": submission_result.get("error"),
    }
    output = manifest_path.with_name("analysis_scf_submission.json")
    record["submission_manifest_path"] = str(output)
    output.write_text(json.dumps(record, indent=2))
    return record


def prepare_adsorption_analysis_scf(
    relaxation_manifest: Any,
    output_directory: str,
    analyses: Optional[List[str]] = None,
    candidate_id: Optional[str] = None,
    kpoint_num: Optional[List[int]] = None,
    incar_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare the best converged UMA candidate for a static VASP SCF."""
    manifest = _load_manifest(relaxation_manifest)
    if manifest.get("artifact_kind") != "ranked_adsorption_relaxations":
        raise ValueError("Expected a ranked_adsorption_relaxations artifact")

    requested = ["bader", "lobster"] if analyses is None else analyses
    if not isinstance(requested, list) or not requested or any(a not in SUPPORTED_ANALYSES for a in requested):
        raise ValueError("analyses must contain only 'bader' and/or 'lobster'")
    requested = list(dict.fromkeys(requested))

    mesh = kpoint_num or [4, 4, 1]
    if (
        not isinstance(mesh, list)
        or len(mesh) != 3
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in mesh)
    ):
        raise ValueError("kpoint_num must contain three positive integers")
    if incar_overrides is not None and not isinstance(incar_overrides, dict):
        raise ValueError("incar_overrides must be an object")

    eligible = []
    for result in manifest.get("results", []):
        diagnostics = result.get("geometry_diagnostics") or {}
        invalid_geometry = any(diagnostics.get(flag) for flag in GEOMETRY_FAILURE_FLAGS)
        path = result.get("final_structure_path")
        if result.get("status") == "converged" and not invalid_geometry and path and Path(path).is_file():
            eligible.append(result)
    if not eligible:
        raise ValueError("No converged, geometrically valid candidate is available")

    eligible.sort(key=lambda item: (float(item["final_energy_eV"]), int(item.get("candidate_index", 0))))
    if candidate_id is None:
        selected = eligible[0]
        policy = "lowest_final_uma_energy"
    else:
        matches = [item for item in eligible if item.get("candidate_id") == candidate_id]
        if not matches:
            raise ValueError(f"Requested candidate is not eligible: {candidate_id}")
        selected, policy = matches[0], "explicit_candidate_id"

    source = Path(selected["final_structure_path"]).expanduser().resolve()
    settings = {**DEFAULT_INCAR, **(incar_overrides or {})}
    for key, expected in (("NSW", 0), ("IBRION", -1), ("LCHARG", True), ("LWAVE", True)):
        if settings.get(key) != expected:
            raise ValueError(f"Static analysis SCF requires {key}={expected}")
    if "bader" in requested and settings.get("LAECHG") is not True:
        raise ValueError("Bader preparation requires LAECHG=true")
    if settings.get("LSORBIT") is not False:
        raise ValueError("SOC is unsupported by this adsorption-analysis preparation stage")

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_relaxation_artifact_id": manifest.get("artifact_id"),
        "candidate_id": selected["candidate_id"],
        "source_structure_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection_policy": policy,
        "uma_final_energy_eV": selected["final_energy_eV"],
        "analyses": requested,
        "incar_tags": settings,
        "kpoint_num": mesh,
    }
    artifact_id = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output = Path(output_directory).expanduser().resolve() / f"adsorption_analysis_scf_{artifact_id}"
    output.mkdir(parents=True, exist_ok=True)
    structure_path = output / f"{selected['candidate_id']}_uma_relaxed.vasp"
    shutil.copy2(source, structure_path)

    response = {
        "success": True,
        "artifact_kind": "adsorption_analysis_scf_preparation",
        "artifact_id": artifact_id,
        "manifest_path": str((output / "analysis_scf_manifest.json").resolve()),
        "selected_candidate_id": selected["candidate_id"],
        "selection_policy": policy,
        "uma_rank": selected.get("rank"),
        "uma_final_energy_eV": selected["final_energy_eV"],
        "global_minimum_claimed": False,
        "structure_path": str(structure_path),
        "requested_analyses": requested,
        "vasp_scf_arguments": {
            "structure_path": str(structure_path),
            "soc": False,
            "incar_tags": settings,
            "kpoint_num": mesh,
        },
        "required_outputs": {
            "bader": ["CHGCAR", "AECCAR0", "AECCAR2"] if "bader" in requested else [],
            "lobster": ["WAVECAR", "POSCAR", "POTCAR"] if "lobster" in requested else [],
        },
        "warnings": (["LOBSTER may require an explicit NBANDS chosen for its basis; this preparation does not infer NBANDS."] if "lobster" in requested else []),
        "submitted": False,
        "provenance": provenance,
    }
    Path(response["manifest_path"]).write_text(json.dumps(response, indent=2))
    return response
