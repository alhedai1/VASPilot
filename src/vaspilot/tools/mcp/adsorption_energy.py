"""Prepare and evaluate method-consistent adsorption-energy calculations."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase.io import read, write


PREPARATION_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
GEOMETRY_FAILURE_FLAGS = (
    "desorbed",
    "penetrated_surface",
    "fragmented",
    "severe_slab_reconstruction",
)
VASP_DEFAULT_SETTINGS = {
    "xc": "PBE",
    "encut_eV": 450.0,
    "pseudopotential_family": "PBE",
    "kpoint_policy": "automatic_density_40",
    "spin_polarized": False,
    "electronic_convergence_eV": 1e-5,
    "ionic_force_convergence_eV_per_A": 0.03,
    "smearing": "gaussian",
    "sigma_eV": 0.05,
    "dipole_correction": True,
    "dipole_direction": 3,
}
ML_DEFAULT_SETTINGS = {"model": "uma-s-1p2", "task": "oc20", "precision": "float32"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(manifest: Any) -> Dict[str, Any]:
    if isinstance(manifest, dict):
        return manifest
    path = Path(manifest).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Relaxation manifest not found: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Relaxation manifest must contain a JSON object")
    return value


def _method_signature(method: str, settings: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"method": method, "settings": settings},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]


def prepare_adsorption_energy_calculations(
    relaxation_manifest: Any,
    clean_slab_structure_path: str,
    adsorbate_structure_path: str,
    output_directory: str,
    top_k: Optional[int] = None,
    method: str = "vasp_dft",
    method_settings: Optional[Dict[str, Any]] = None,
    isolated_adsorbate_box_size: float = 20.0,
    adsorbate_charge: int = 0,
    adsorbate_multiplicity: int = 1,
    reject_geometry_warnings: bool = True,
) -> Dict[str, Any]:
    """Select relaxed candidates and prepare consistent calculation inputs."""
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer or null")
    if method not in {"vasp_dft", "ml_screening"}:
        raise ValueError("method must be 'vasp_dft' or 'ml_screening'")
    if not isinstance(isolated_adsorbate_box_size, (int, float)) or isolated_adsorbate_box_size <= 5:
        raise ValueError("isolated_adsorbate_box_size must be greater than 5 A")
    if method_settings is not None and not isinstance(method_settings, dict):
        raise ValueError("method_settings must be an object")
    settings = {
        **(VASP_DEFAULT_SETTINGS if method == "vasp_dft" else ML_DEFAULT_SETTINGS),
        **(method_settings or {}),
    }
    if not isinstance(adsorbate_charge, int) or isinstance(adsorbate_charge, bool):
        raise ValueError("adsorbate_charge must be an integer")
    if not isinstance(adsorbate_multiplicity, int) or isinstance(adsorbate_multiplicity, bool) or adsorbate_multiplicity < 1:
        raise ValueError("adsorbate_multiplicity must be a positive integer")
    if method == "vasp_dft" and adsorbate_charge and "nelect" not in settings:
        raise ValueError("Charged VASP adsorbates require explicit method_settings.nelect")
    if method == "vasp_dft" and adsorbate_multiplicity > 1 and not settings["spin_polarized"]:
        raise ValueError("Adsorbate multiplicity greater than one requires spin_polarized=true")

    manifest = _load_manifest(relaxation_manifest)
    slab_path = Path(clean_slab_structure_path).expanduser()
    adsorbate_path = Path(adsorbate_structure_path).expanduser()
    for label, path in (("Clean slab", slab_path), ("Adsorbate", adsorbate_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} structure not found: {path}")

    selected, rejected = [], []
    for result in manifest.get("results", []):
        reasons = []
        if result.get("status") != "converged":
            reasons.append("not_converged")
        diagnostics = result.get("geometry_diagnostics", {})
        if reject_geometry_warnings:
            reasons.extend(flag for flag in GEOMETRY_FAILURE_FLAGS if diagnostics.get(flag))
        structure_path = result.get("final_structure_path")
        if not structure_path or not Path(structure_path).is_file():
            reasons.append("relaxed_structure_missing")
        if reasons:
            rejected.append({"candidate_id": result.get("candidate_id"), "reasons": reasons})
        else:
            selected.append(result)
    selected.sort(key=lambda item: (item.get("rank") is None, item.get("rank") or 0))
    if top_k is not None:
        selected = selected[:top_k]
    if not selected:
        raise ValueError("No converged, geometrically acceptable candidates are available")

    slab = read(slab_path)
    adsorbate = read(adsorbate_path)
    candidate_atoms = [read(item["final_structure_path"]) for item in selected]
    expected_symbols = sorted(slab.get_chemical_symbols() + adsorbate.get_chemical_symbols())
    for item, atoms in zip(selected, candidate_atoms):
        if sorted(atoms.get_chemical_symbols()) != expected_symbols:
            raise ValueError(f"Candidate {item['candidate_id']} is not clean slab plus adsorbate")
        if not np.allclose(atoms.cell, slab.cell, atol=1e-6):
            raise ValueError(f"Candidate {item['candidate_id']} cell differs from clean slab cell")

    method_signature = _method_signature(method, settings)
    provenance = {
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "relaxation_artifact_id": manifest.get("artifact_id"),
        "clean_slab_sha256": _sha256(slab_path),
        "adsorbate_sha256": _sha256(adsorbate_path),
        "candidate_sha256": [_sha256(Path(item["final_structure_path"])) for item in selected],
        "top_k": top_k,
        "method": method,
        "method_settings": settings,
        "method_signature": method_signature,
        "isolated_adsorbate_box_size_A": float(isolated_adsorbate_box_size),
        "adsorbate_charge": adsorbate_charge,
        "adsorbate_multiplicity": adsorbate_multiplicity,
        "reject_geometry_warnings": reject_geometry_warnings,
    }
    artifact_id = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output = Path(output_directory).expanduser().resolve() / f"adsorption_energy_inputs_{artifact_id}"
    output.mkdir(parents=True, exist_ok=True)

    clean_output = output / "clean_slab.vasp"
    write(clean_output, slab, format="vasp", direct=True, sort=False)
    adsorbate.set_cell([isolated_adsorbate_box_size] * 3)
    adsorbate.center()
    adsorbate.set_pbc(True)
    adsorbate_output = output / "isolated_adsorbate.vasp"
    write(adsorbate_output, adsorbate, format="vasp", direct=True, sort=False)

    calculations = [
        {"calculation_id": f"{artifact_id}_clean_slab", "role": "clean_slab", "structure_path": str(clean_output), "method_signature": method_signature},
        {"calculation_id": f"{artifact_id}_isolated_adsorbate", "role": "isolated_adsorbate", "structure_path": str(adsorbate_output), "method_signature": method_signature, "charge": adsorbate_charge, "multiplicity": adsorbate_multiplicity},
    ]
    for item, atoms in zip(selected, candidate_atoms):
        candidate_output = output / f"{item['candidate_id']}.vasp"
        write(candidate_output, atoms, format="vasp", direct=True, sort=False)
        calculations.append({"calculation_id": f"{artifact_id}_{item['candidate_id']}", "role": "adsorbed_candidate", "candidate_id": item["candidate_id"], "screening_rank": item.get("rank"), "structure_path": str(candidate_output), "method_signature": method_signature})

    response = {
        "success": True,
        "artifact_kind": "adsorption_energy_calculation_set",
        "artifact_id": artifact_id,
        "output_directory": str(output),
        "method": method,
        "method_settings": settings,
        "method_signature": method_signature,
        "selected_candidate_count": len(selected),
        "rejected_candidates": rejected,
        "calculations": calculations,
        "provenance": provenance,
        "submitted": False,
        "submission_note": "Submit each calculation with the existing calculator tool, then pass completed energy records to calculate_adsorption_energies.",
    }
    manifest_path = output / "calculation_manifest.json"
    response["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(response, indent=2))
    return response


def calculate_adsorption_energies(
    candidate_energy_records: List[Dict[str, Any]],
    clean_slab_energy_record: Dict[str, Any],
    adsorbate_energy_record: Dict[str, Any],
    output_directory: str,
    corrections: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Validate completed records and calculate electronic adsorption energies."""
    if not candidate_energy_records:
        raise ValueError("At least one candidate energy record is required")
    records = [clean_slab_energy_record, adsorbate_energy_record, *candidate_energy_records]
    required = {"energy_eV", "converged", "method_signature"}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Energy record is missing fields: {sorted(missing)}")
        if record["converged"] is not True:
            raise ValueError(f"Energy record is not converged: {record.get('calculation_id', 'unknown')}")
        if not isinstance(record["energy_eV"], (int, float)):
            raise ValueError("energy_eV must be numeric")
    signatures = {record["method_signature"] for record in records}
    if len(signatures) != 1:
        raise ValueError("All adsorption-energy terms must use the same method_signature")
    if len({record.get("candidate_id") for record in candidate_energy_records}) != len(candidate_energy_records):
        raise ValueError("Candidate energy records must have unique candidate_id values")

    correction_map = corrections or {}
    if not isinstance(correction_map, dict):
        raise ValueError("corrections must be an object")
    slab_energy = float(clean_slab_energy_record["energy_eV"])
    adsorbate_energy = float(adsorbate_energy_record["energy_eV"])
    results = []
    for record in candidate_energy_records:
        candidate_id = record.get("candidate_id")
        if not candidate_id:
            raise ValueError("Every candidate energy record requires candidate_id")
        components = correction_map.get(candidate_id, {})
        if not isinstance(components, dict) or not all(isinstance(value, (int, float)) for value in components.values()):
            raise ValueError(f"Corrections for {candidate_id} must be numeric")
        electronic = float(record["energy_eV"]) - slab_energy - adsorbate_energy
        correction = float(sum(components.values()))
        results.append({
            "candidate_id": candidate_id,
            "combined_energy_eV": float(record["energy_eV"]),
            "clean_slab_energy_eV": slab_energy,
            "isolated_adsorbate_energy_eV": adsorbate_energy,
            "electronic_adsorption_energy_eV": electronic,
            "correction_components_eV": components,
            "total_correction_eV": correction,
            "corrected_adsorption_energy_eV": electronic + correction,
            "rank": None,
            "calculation_path": record.get("calculation_path"),
        })
    ranked = sorted(results, key=lambda item: (item["corrected_adsorption_energy_eV"], item["candidate_id"]))
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank

    provenance = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "method_signature": signatures.pop(),
        "candidate_calculation_ids": [record.get("calculation_id") for record in candidate_energy_records],
        "clean_slab_calculation_id": clean_slab_energy_record.get("calculation_id"),
        "adsorbate_calculation_id": adsorbate_energy_record.get("calculation_id"),
        "corrections": correction_map,
        "energy_convention": "E_adsorbed - E_clean_slab - E_isolated_adsorbate",
    }
    artifact_id = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    response = {
        "success": True,
        "artifact_kind": "ranked_adsorption_energies",
        "artifact_id": artifact_id,
        "method_signature": provenance["method_signature"],
        "energy_convention": provenance["energy_convention"],
        "corrections_applied": bool(correction_map),
        "best_candidate_id": ranked[0]["candidate_id"],
        "results": ranked,
        "provenance": provenance,
    }
    manifest_path = output / f"adsorption_energies_{artifact_id}.json"
    response["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(response, indent=2))
    return response
