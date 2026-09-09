"""Generate reproducible adsorbate-slab candidates with FAIR Chemistry."""

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path


def _json_safe(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class CandidateBudgetExceeded(ValueError):
    """Raised when a request exceeds its explicit candidate budget."""


def _wrap_in_plane(atoms) -> None:
    scaled = atoms.get_scaled_positions(wrap=False)
    scaled[:, :2] %= 1.0
    atoms.set_scaled_positions(scaled)


def _wrap_candidate_in_plane(
    atoms, slab_atom_count: int, binding_atom_indices: list[int]
) -> None:
    import numpy as np

    scaled = atoms.get_scaled_positions(wrap=False)
    scaled[:slab_atom_count, :2] %= 1.0
    binding_indices = slab_atom_count + np.asarray(binding_atom_indices)
    shift = np.floor(scaled[binding_indices, :2].mean(axis=0))
    scaled[slab_atom_count:, :2] -= shift
    atoms.set_scaled_positions(scaled)


def _geometry_diagnostics(atoms, slab_atom_count: int, surface_normal) -> dict:
    import numpy as np

    slab_indices = np.arange(slab_atom_count)
    adsorbate_indices = np.arange(slab_atom_count, len(atoms))
    heights = atoms.positions @ surface_normal
    distances = atoms.get_distances(
        adsorbate_indices[:, None], slab_indices[None, :], mic=True
    )
    relative_heights = heights[adsorbate_indices] - heights[slab_indices].max()
    return {
        "minimum_adsorbate_slab_distance_A": float(distances.min()),
        "adsorbate_height_range_above_top_surface_A": [
            float(relative_heights.min()),
            float(relative_heights.max()),
        ],
        "all_adsorbate_atoms_above_top_surface": bool(np.all(relative_heights > 0)),
        "slab_in_plane_coordinates_wrapped": True,
        "adsorbate_periodic_image_selected_as_rigid_unit": True,
    }


def validate_generation_size(
    num_sites: int,
    num_orientations_per_site: int,
    max_generated_candidates: int,
) -> None:
    values = {
        "num_sites": num_sites,
        "num_orientations_per_site": num_orientations_per_site,
        "max_generated_candidates": max_generated_candidates,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    requested = num_sites * num_orientations_per_site
    if requested > max_generated_candidates:
        raise CandidateBudgetExceeded(
            f"Requested up to {requested} candidates "
            f"({num_sites} sites × {num_orientations_per_site} orientations), "
            f"which exceeds max_generated_candidates={max_generated_candidates}"
        )

def generate_candidates(
    slab_path: Path,
    adsorbate_path: Path,
    binding_atom_indices: list[int],
    num_sites: int,
    num_orientations_per_site: int,
    max_generated_candidates: int = 1000,
    placement_mode: str = "heuristic",
    random_seed: int = 0,
    interstitial_gap: float = 0.1,
    surface_layer_tolerance: float = 0.5,
) -> dict:
    import numpy as np
    from ase.io import read, write
    from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Slab

    validate_generation_size(
        num_sites,
        num_orientations_per_site,
        max_generated_candidates,
    )

    if placement_mode not in {"heuristic", "random", "random_site_heuristic_placement"}:
        raise ValueError("Unsupported placement_mode")
    if not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    if not 0 <= interstitial_gap < 5:
        raise ValueError("interstitial_gap must be at least 0 and less than 5 A")
    if not 0 < surface_layer_tolerance <= 2:
        raise ValueError("surface_layer_tolerance must be greater than 0 and at most 2 A")
    if not slab_path.is_file():
        raise FileNotFoundError(f"Slab structure not found: {slab_path}")
    if not adsorbate_path.is_file():
        raise FileNotFoundError(f"Adsorbate structure not found: {adsorbate_path}")

    slab_atoms = read(slab_path)
    adsorbate_atoms = read(adsorbate_path)
    if not binding_atom_indices:
        raise ValueError("At least one binding atom index is required")
    if any(index < 0 or index >= len(adsorbate_atoms) for index in binding_atom_indices):
        raise ValueError("A binding atom index is outside the adsorbate atom range")
    if not slab_atoms.constraints:
        raise ValueError(
            "The slab has no fixed-atom constraint; use a slab produced by build_surface"
        )

    slab_atoms.set_pbc([True, True, True])
    _wrap_in_plane(slab_atoms)
    surface_normal = np.cross(slab_atoms.cell[0], slab_atoms.cell[1])
    surface_normal /= np.linalg.norm(surface_normal)
    heights = np.dot(slab_atoms.positions, surface_normal)
    tags = np.zeros(len(slab_atoms), dtype=int)
    tags[heights >= heights.max() - surface_layer_tolerance] = 1
    slab_atoms.set_tags(tags)

    np.random.seed(random_seed)
    slab = Slab(
        slab_atoms=slab_atoms,
        millers=None,
        shift=None,
        top=True,
    )
    adsorbate = Adsorbate(
        adsorbate_atoms=adsorbate_atoms,
        adsorbate_binding_indices=binding_atom_indices,
    )
    configuration = AdsorbateSlabConfig(
        slab=slab,
        adsorbate=adsorbate,
        num_sites=num_sites,
        num_augmentations_per_site=num_orientations_per_site,
        interstitial_gap=interstitial_gap,
        mode=placement_mode,
    )

    available_site_count = len(configuration.sites)
    selected_site_count = min(num_sites, available_site_count)
    candidate_limit = selected_site_count * num_orientations_per_site
    parameters = {
        "placement_mode": placement_mode,
        "requested_num_sites": num_sites,
        "num_orientations_per_site": num_orientations_per_site,
        "max_generated_candidates": max_generated_candidates,
        "requested_candidate_upper_bound": (
            num_sites * num_orientations_per_site
        ),
        "random_seed": random_seed,
        "interstitial_gap_A": interstitial_gap,
        "surface_layer_tolerance_A": surface_layer_tolerance,
        "surface_side": "top",
    }
    provenance = {
        "generator_schema_version": 3,
        "slab_sha256": hashlib.sha256(slab_path.read_bytes()).hexdigest(),
        "adsorbate_sha256": hashlib.sha256(adsorbate_path.read_bytes()).hexdigest(),
        "binding_atom_indices": binding_atom_indices,
        "parameters": parameters,
    }
    candidate_set_id = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_directory = slab_path.with_name(
        f"{slab_path.stem}_{adsorbate_path.stem}_candidates_{candidate_set_id}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    candidates = []
    for index, atoms in enumerate(configuration.atoms_list[:candidate_limit]):
        atoms.set_pbc([True, True, True])
        _wrap_candidate_in_plane(atoms, len(slab_atoms), binding_atom_indices)
        candidate_id = f"candidate_{index:03d}"
        structure_path = output_directory / f"{candidate_id}.vasp"
        preview_path = output_directory / f"{candidate_id}.xyz"
        metadata_path = output_directory / f"{candidate_id}.json"
        write(structure_path, atoms, format="vasp", direct=True, sort=False)
        write(preview_path, atoms, format="xyz")
        placement_metadata = (
            configuration.metadata_list[index]
            if index < len(configuration.metadata_list)
            else {}
        )
        adsorbate_indices = list(range(len(slab_atoms), len(atoms)))
        geometry = _geometry_diagnostics(atoms, len(slab_atoms), surface_normal)
        metadata_path.write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "candidate_set_id": candidate_set_id,
                    "slab_atom_count": len(slab_atoms),
                    "adsorbate_atom_indices": adsorbate_indices,
                    "binding_atom_indices_within_adsorbate": binding_atom_indices,
                    "placement_metadata": _json_safe(placement_metadata),
                    "geometry_diagnostics": geometry,
                },
                separators=(",", ":"),
            )
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_set_id": candidate_set_id,
                "structure_path": str(structure_path.resolve()),
                "preview_path": str(preview_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "num_atoms": len(atoms),
                "adsorbate_atom_indices": adsorbate_indices,
                "placement_metadata": _json_safe(placement_metadata),
                "geometry_diagnostics": geometry,
            }
        )

    if not candidates:
        raise RuntimeError("FAIR Chemistry generated no adsorption candidates")

    response = {
        "success": True,
        "error": None,
        "artifact_kind": "adsorption_candidate_set",
        "artifact_id": candidate_set_id,
        "slab_structure_path": str(slab_path.resolve()),
        "adsorbate_structure_path": str(adsorbate_path.resolve()),
        "binding_atom_indices": binding_atom_indices,
        "binding_indexing": "zero_based",
        "candidate_count": len(candidates),
        "candidate_directory": str(output_directory.resolve()),
        "candidates": candidates,
        "generation_metadata": {
            "engine": "FAIR Chemistry AdsorbateSlabConfig",
            **parameters,
            "available_site_count": available_site_count,
            "selected_site_count": selected_site_count,
            "candidate_count_semantics": "up_to_requested_sites_times_orientations",
            "surface_atom_count": int(np.count_nonzero(tags == 1)),
            **provenance,
            "relaxed": False,
            "energy_ranked": False,
        },
    }
    manifest_path = output_directory / "candidate_manifest.json"
    response["manifest_path"] = str(manifest_path.resolve())
    manifest_path.write_text(json.dumps(response, indent=2, allow_nan=False))
    return response


def main() -> int:
    result = None
    try:
        if len(sys.argv) != 11:
            raise ValueError(
                "Usage: adsorption_candidate_runner.py SLAB ADSORBATE "
                "BINDING_INDICES NUM_SITES NUM_ORIENTATIONS "
                "MAX_GENERATED_CANDIDATES MODE SEED GAP LAYER_TOLERANCE"
            )
        binding_indices = [
            int(value) for value in sys.argv[3].split(",") if value.strip()
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            result = generate_candidates(
                slab_path=Path(sys.argv[1]).expanduser(),
                adsorbate_path=Path(sys.argv[2]).expanduser(),
                binding_atom_indices=binding_indices,
                num_sites=int(sys.argv[4]),
                num_orientations_per_site=int(sys.argv[5]),
                max_generated_candidates=int(sys.argv[6]),
                placement_mode=sys.argv[7],
                random_seed=int(sys.argv[8]),
                interstitial_gap=float(sys.argv[9]),
                surface_layer_tolerance=float(sys.argv[10]),
            )
        exit_code = 0
    except CandidateBudgetExceeded as exc:
        result = {
            "success": False,
            "error_code": "candidate_budget_exceeded",
            "error": str(exc),
            "retryable": False,
        }
        exit_code = 1
    except (ValueError, FileNotFoundError) as exc:
        result = {
            "success": False,
            "error_code": "invalid_candidate_generation_input",
            "error": str(exc),
            "retryable": False,
        }
        exit_code = 1
    except Exception as exc:
        result = {
            "success": False,
            "error_code": "candidate_generation_failed",
            "error": str(exc),
            "retryable": False,
        }
        exit_code = 1

    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
