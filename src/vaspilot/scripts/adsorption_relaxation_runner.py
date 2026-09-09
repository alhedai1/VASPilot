"""Relax and rank validated adsorption candidates with FAIR Chemistry."""

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

SCHEMA_VERSION = 2


def validate_request(request: dict) -> dict:
    defaults = {"calculator_backend": "fairchem", "model": "uma-s-1p2", "task": "oc20", "device": "auto", "precision": "float32", "optimizer": "LBFGS", "fmax": 0.05, "max_steps": 200, "desorption_distance_A": 4.0, "penetration_depth_A": 1.0, "severe_slab_displacement_A": 2.0}
    config = {**defaults, **request}
    paths, candidate_set = config.get("candidate_structure_paths"), config.get("candidate_set")
    manifest_path = config.get("candidate_manifest_path")
    if manifest_path is not None:
        if paths or candidate_set is not None:
            raise ValueError("Provide exactly one candidate input")
        path = Path(manifest_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Candidate manifest not found: {path}")
        candidate_set = json.loads(path.read_text())
    if candidate_set is not None:
        if paths:
            raise ValueError("Provide exactly one candidate input")
        if not isinstance(candidate_set, dict) or not isinstance(candidate_set.get("candidates"), list):
            raise ValueError("candidate_set must contain a candidates list")
        paths = [item.get("structure_path") for item in candidate_set["candidates"]]
        config["candidate_set_id"] = candidate_set.get("artifact_id")
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths):
        raise ValueError("At least one candidate structure path is required")
    if len(paths) != len(set(paths)):
        raise ValueError("Candidate structure paths must be unique")
    config["candidate_structure_paths"] = paths
    if config["calculator_backend"] != "fairchem":
        raise ValueError("calculator_backend must be 'fairchem'")
    if not all(isinstance(config[key], str) and config[key] for key in ("model", "task")):
        raise ValueError("model and task must be non-empty strings")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if config["precision"] not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")
    if config["optimizer"] not in {"LBFGS", "BFGS", "FIRE"}:
        raise ValueError("optimizer must be 'LBFGS', 'BFGS', or 'FIRE'")
    if not isinstance(config["fmax"], (int, float)) or not 0 < config["fmax"] <= 1:
        raise ValueError("fmax must be greater than 0 and at most 1 eV/A")
    if not isinstance(config["max_steps"], int) or isinstance(config["max_steps"], bool) or not 1 <= config["max_steps"] <= 2000:
        raise ValueError("max_steps must be between 1 and 2000")
    for key in ("desorption_distance_A", "penetration_depth_A", "severe_slab_displacement_A"):
        if not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive number")
    return config


def rank_converged(results: list[dict]) -> list[dict]:
    ranked = sorted((item for item in results if item["status"] == "converged"), key=lambda item: (item["final_energy_eV"], item["candidate_index"]))
    if ranked:
        best = ranked[0]["final_energy_eV"]
        for rank, item in enumerate(ranked, 1):
            item["rank"], item["energy_relative_to_best_eV"] = rank, item["final_energy_eV"] - best
    return ranked


def geometry_diagnostics(initial, final, adsorbate_indices: list[int], config: dict) -> dict:
    import numpy as np
    from ase.data import covalent_radii

    if not adsorbate_indices:
        return {"available": False, "reason": "adsorbate atom indices unavailable"}
    ads, ads_set = np.asarray(adsorbate_indices, dtype=int), set(adsorbate_indices)
    slab = np.asarray([index for index in range(len(final)) if index not in ads_set], dtype=int)
    normal = np.cross(final.cell[0], final.cell[1]); normal /= np.linalg.norm(normal)
    heights = final.positions @ normal
    distances = final.get_distances(ads[:, None], slab[None, :], mic=True)
    relative_heights = heights[ads] - heights[slab].max()
    displacement_vectors = final.positions[slab] - initial.positions[slab]
    fractional_displacements = np.linalg.solve(final.cell.T, displacement_vectors.T).T
    fractional_displacements -= np.round(fractional_displacements)
    displacement_vectors = fractional_displacements @ final.cell
    slab_displacements = np.linalg.norm(displacement_vectors, axis=1)
    broken_bonds = []
    for offset, left in enumerate(ads):
        for right in ads[offset + 1:]:
            initial_distance = initial.get_distance(left, right, mic=True)
            cutoff = 1.25 * (covalent_radii[initial.numbers[left]] + covalent_radii[initial.numbers[right]])
            if initial_distance <= cutoff:
                final_distance = final.get_distance(left, right, mic=True)
                if final_distance > max(1.75 * initial_distance, 1.5 * cutoff):
                    broken_bonds.append([int(left), int(right)])
    minimum_distance, minimum_height = float(distances.min()), float(relative_heights.min())
    maximum_displacement = float(slab_displacements.max()) if len(slab) else 0.0
    return {"available": True, "minimum_adsorbate_slab_distance_A": minimum_distance, "adsorbate_height_range_above_top_surface_A": [minimum_height, float(relative_heights.max())], "desorbed": minimum_distance > config["desorption_distance_A"], "penetrated_surface": minimum_height < -config["penetration_depth_A"], "fragmented": bool(broken_bonds), "broken_adsorbate_bonds": broken_bonds, "maximum_slab_atom_displacement_A": maximum_displacement, "severe_slab_reconstruction": maximum_displacement > config["severe_slab_displacement_A"]}


def _metadata(path: Path, atom_count: int, expected_set_id: str | None) -> tuple[dict, list[int]]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}, []
    metadata = json.loads(metadata_path.read_text())
    if expected_set_id and metadata.get("candidate_set_id") != expected_set_id:
        raise ValueError(f"Candidate {path} does not belong to candidate set {expected_set_id}")
    indices = metadata.get("adsorbate_atom_indices", [])
    if not isinstance(indices, list) or any(not isinstance(index, int) or index < 0 or index >= atom_count for index in indices):
        raise ValueError(f"Candidate metadata has invalid adsorbate indices: {metadata_path}")
    return metadata, indices


def relax_candidates(request: dict) -> dict:
    import numpy as np
    import torch
    from ase.io import read, write
    from ase.optimize import BFGS, FIRE, LBFGS
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    config = validate_request(request)
    paths = [Path(path).expanduser() for path in config["candidate_structure_paths"]]
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(f"Candidate structure not found: {path}")
    reference = read(paths[0])
    for path in paths[1:]:
        candidate = read(path)
        if candidate.get_chemical_symbols() != reference.get_chemical_symbols(): raise ValueError("All candidates must have identical atom ordering and composition")
        if not np.allclose(candidate.cell, reference.cell, atol=1e-6): raise ValueError("All candidates must have identical simulation cells")
    requested_device = config["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    if device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA was requested but is not available")
    predictor = pretrained_mlip.get_predict_unit(config["model"], device=device, inference_settings=InferenceSettings(compile=False, base_precision_dtype=config["precision"]))
    calculator = FAIRChemCalculator(predictor, task_name=config["task"])
    optimizer_class = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}[config["optimizer"]]
    config_keys = ("calculator_backend", "model", "task", "device", "precision", "optimizer", "fmax", "max_steps", "desorption_distance_A", "penetration_depth_A", "severe_slab_displacement_A")
    provenance = {"relaxation_schema_version": SCHEMA_VERSION, "candidate_set_id": config.get("candidate_set_id"), "candidate_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths], "configuration": {key: config[key] for key in config_keys}}
    artifact_id = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    output_directory = paths[0].parent / f"relaxations_{artifact_id}"; output_directory.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for candidate_index, path in enumerate(paths):
        candidate_id = path.stem
        artifacts = {"trajectory_path": output_directory / f"{candidate_id}.traj", "trajectory_extxyz_path": output_directory / f"{candidate_id}.extxyz", "logfile_path": output_directory / f"{candidate_id}.log", "final_structure_path": output_directory / f"{candidate_id}_relaxed.vasp"}
        try:
            atoms = read(path)
            if not atoms.constraints: raise ValueError("Candidate does not preserve fixed slab atoms")
            initial = atoms.copy()
            metadata, adsorbate_indices = _metadata(path, len(atoms), config.get("candidate_set_id"))
            if adsorbate_indices:
                tags = np.zeros(len(atoms), dtype=int); tags[adsorbate_indices] = 2; atoms.set_tags(tags)
            atoms.set_pbc(True); atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy())
            optimizer = optimizer_class(atoms, trajectory=str(artifacts["trajectory_path"]), logfile=str(artifacts["logfile_path"]))
            converged = bool(optimizer.run(fmax=config["fmax"], steps=config["max_steps"]))
            final_energy = float(atoms.get_potential_energy()); max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            write(artifacts["final_structure_path"], atoms, format="vasp", direct=True, sort=False)
            frames = read(artifacts["trajectory_path"], index=":"); write(artifacts["trajectory_extxyz_path"], frames, format="extxyz")
            results.append({"candidate_id": candidate_id, "candidate_set_id": metadata.get("candidate_set_id"), "candidate_index": candidate_index, "status": "converged" if converged else "not_converged", "initial_structure_path": str(path.resolve()), **{key: str(value.resolve()) for key, value in artifacts.items()}, "initial_energy_eV": initial_energy, "final_energy_eV": final_energy, "energy_relative_to_best_eV": None, "rank": None, "steps": int(optimizer.nsteps), "trajectory_frames": len(frames), "converged": converged, "final_max_force_eV_per_A": max_force, "max_atomic_force_eV_per_A": max_force, "geometry_diagnostics": geometry_diagnostics(initial, atoms, adsorbate_indices, config)})
        except Exception as exc:
            failures.append({"candidate_id": candidate_id, "candidate_index": candidate_index, "status": "failed", "initial_structure_path": str(path.resolve()), "error_type": type(exc).__name__, "error": str(exc), **{key: str(value.resolve()) if value.is_file() else None for key, value in artifacts.items()}})
    ranked = rank_converged(results)
    complete_success = not failures and len(ranked) == len(paths)
    response = {"success": bool(results), "complete_success": complete_success, "error": None if results else "All candidate relaxations failed", "artifact_kind": "ranked_adsorption_relaxations", "artifact_id": artifact_id, "output_directory": str(output_directory.resolve()), "provenance": provenance, "calculator_backend": config["calculator_backend"], "model": config["model"], "task": config["task"], "requested_device": requested_device, "device": device, "precision": config["precision"], "optimizer": config["optimizer"], "cell_relaxed": False, "constraints_preserved": True, "fmax_eV_per_A": config["fmax"], "max_steps": config["max_steps"], "candidate_count": len(paths), "processed_count": len(results), "converged_count": len(ranked), "nonconverged_count": len(results) - len(ranked), "failed_count": len(failures), "all_candidates_converged": complete_success, "best_candidate_id": ranked[0]["candidate_id"] if ranked else None, "best_final_structure_path": ranked[0]["final_structure_path"] if ranked else None, "best_trajectory_path": ranked[0]["trajectory_path"] if ranked else None, "best_trajectory_extxyz_path": ranked[0]["trajectory_extxyz_path"] if ranked else None, "ranking_basis": "lowest_final_total_energy_among_converged_same-composition_same-cell_candidates", "global_minimum_claimed": False, "results": sorted(results, key=lambda item: (item["rank"] is None, item["rank"] or item["candidate_index"])), "failures": failures}
    manifest_path = output_directory / "relaxation_manifest.json"
    response["manifest_path"] = str(manifest_path.resolve())
    manifest_path.write_text(json.dumps(response, indent=2, allow_nan=False))
    return response


def main() -> int:
    try:
        if len(sys.argv) != 2: raise ValueError("Usage: adsorption_relaxation_runner.py REQUEST_JSON")
        with contextlib.redirect_stdout(io.StringIO()): result = relax_candidates(json.loads(sys.argv[1]))
        exit_code = 0 if result.get("success") else 1
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        result, exit_code = {"success": False, "error_code": "invalid_relaxation_input", "error": str(exc), "retryable": False}, 1
    except Exception as exc:
        result, exit_code = {"success": False, "error_code": "relaxation_failed", "error": str(exc), "retryable": False}, 1
    print(json.dumps(result, separators=(",", ":"), allow_nan=False)); return exit_code


if __name__ == "__main__": raise SystemExit(main())
