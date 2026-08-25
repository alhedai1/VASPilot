"""Relax adsorption candidates with UMA/OC20 and save every trajectory."""

import contextlib
import io
import json
import sys
from pathlib import Path


MODEL = "uma-s-1p2"
TASK = "oc20"


def relax_candidates(
    candidate_paths: list[Path],
    fmax: float,
    max_steps: int,
) -> dict:
    import numpy as np
    import torch
    from ase.io import read, write
    from ase.optimize import LBFGS
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    if not candidate_paths:
        raise ValueError("At least one candidate structure path is required")
    if not 0 < fmax <= 1:
        raise ValueError("fmax must be greater than 0 and at most 1 eV/A")
    if not 1 <= max_steps <= 2000:
        raise ValueError("max_steps must be between 1 and 2000")
    for path in candidate_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Candidate structure not found: {path}")

    reference = read(candidate_paths[0])
    reference_symbols = reference.get_chemical_symbols()
    reference_cell = np.asarray(reference.cell)
    for path in candidate_paths[1:]:
        candidate = read(path)
        if candidate.get_chemical_symbols() != reference_symbols:
            raise ValueError("All candidates must have identical atom ordering and composition")
        if not np.allclose(candidate.cell, reference_cell, atol=1e-6):
            raise ValueError("All candidates must have identical simulation cells")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = pretrained_mlip.get_predict_unit(
        MODEL,
        device=device,
        inference_settings=InferenceSettings(compile=False),
    )
    calculator = FAIRChemCalculator(predictor, task_name=TASK)

    results = []
    failures = []
    fmax_label = format(fmax, ".6g").replace(".", "p")
    for candidate_index, path in enumerate(candidate_paths):
        candidate_id = path.stem
        output_directory = path.parent / (
            f"relaxations_fmax{fmax_label}_steps{max_steps}"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_directory / f"{candidate_id}.traj"
        trajectory_extxyz_path = output_directory / f"{candidate_id}.extxyz"
        logfile_path = output_directory / f"{candidate_id}.log"
        final_structure_path = output_directory / f"{candidate_id}_relaxed.vasp"

        try:
            atoms = read(path)
            if not atoms.constraints:
                raise ValueError("Candidate does not preserve fixed slab atoms")
            metadata_path = path.with_suffix(".json")
            if metadata_path.is_file():
                candidate_metadata = json.loads(metadata_path.read_text())
                tags = np.zeros(len(atoms), dtype=int)
                tags[candidate_metadata["adsorbate_atom_indices"]] = 2
                atoms.set_tags(tags)
            atoms.set_pbc([True, True, True])
            atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy())

            optimizer = LBFGS(
                atoms,
                trajectory=str(trajectory_path),
                logfile=str(logfile_path),
            )
            converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
            final_energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            max_force = (
                float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
            )
            write(final_structure_path, atoms, format="vasp", direct=True, sort=False)
            frames = read(trajectory_path, index=":")
            write(trajectory_extxyz_path, frames, format="extxyz")

            results.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "initial_structure_path": str(path.resolve()),
                    "final_structure_path": str(final_structure_path.resolve()),
                    "trajectory_path": str(trajectory_path.resolve()),
                    "trajectory_extxyz_path": str(trajectory_extxyz_path.resolve()),
                    "logfile_path": str(logfile_path.resolve()),
                    "initial_energy_eV": initial_energy,
                    "final_energy_eV": final_energy,
                    "energy_relative_to_best_eV": None,
                    "steps": int(optimizer.nsteps),
                    "trajectory_frames": len(frames),
                    "converged": converged,
                    "max_atomic_force_eV_per_A": max_force,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "initial_structure_path": str(path.resolve()),
                    "error": str(exc),
                }
            )

    if not results:
        return {
            "success": False,
            "error": "All candidate relaxations failed",
            "results": [],
            "failures": failures,
        }

    ranked = sorted(results, key=lambda result: result["final_energy_eV"])
    best_energy = ranked[0]["final_energy_eV"]
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
        result["energy_relative_to_best_eV"] = (
            result["final_energy_eV"] - best_energy
        )

    return {
        "success": True,
        "error": None,
        "artifact_kind": "ranked_adsorption_relaxations",
        "model": MODEL,
        "task": TASK,
        "device": device,
        "optimizer": "LBFGS",
        "cell_relaxed": False,
        "constraints_preserved": True,
        "fmax_eV_per_A": fmax,
        "max_steps": max_steps,
        "candidate_count": len(candidate_paths),
        "successful_count": len(results),
        "failed_count": len(failures),
        "all_successful_candidates_converged": all(
            result["converged"] for result in results
        ),
        "best_candidate_id": ranked[0]["candidate_id"],
        "best_final_structure_path": ranked[0]["final_structure_path"],
        "best_trajectory_path": ranked[0]["trajectory_path"],
        "best_trajectory_extxyz_path": ranked[0]["trajectory_extxyz_path"],
        "ranking_basis": "lowest_final_total_energy_same_composition_and_cell",
        "global_minimum_claimed": False,
        "results": ranked,
        "failures": failures,
    }


def main() -> int:
    result = None
    try:
        if len(sys.argv) != 4:
            raise ValueError(
                "Usage: adsorption_relaxation_runner.py "
                "CANDIDATE_PATHS_JSON FMAX MAX_STEPS"
            )
        candidate_paths = [
            Path(path).expanduser() for path in json.loads(sys.argv[1])
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            result = relax_candidates(
                candidate_paths=candidate_paths,
                fmax=float(sys.argv[2]),
                max_steps=int(sys.argv[3]),
            )
        exit_code = 0 if result.get("success") else 1
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
        exit_code = 1

    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
