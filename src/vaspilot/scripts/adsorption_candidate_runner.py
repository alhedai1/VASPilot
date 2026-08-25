"""Generate reproducible adsorbate-slab candidates with FAIR Chemistry."""

import contextlib
import io
import json
import sys
from pathlib import Path


RANDOM_SEED = 0
PLACEMENT_MODE = "heuristic"
SURFACE_LAYER_TOLERANCE_A = 0.5


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


def generate_candidates(
    slab_path: Path,
    adsorbate_path: Path,
    binding_atom_indices: list[int],
    num_sites: int,
    num_orientations_per_site: int,
) -> dict:
    import numpy as np
    from ase.io import read, write
    from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Slab

    if not slab_path.is_file():
        raise FileNotFoundError(f"Slab structure not found: {slab_path}")
    if not adsorbate_path.is_file():
        raise FileNotFoundError(f"Adsorbate structure not found: {adsorbate_path}")
    if num_sites < 1 or num_sites > 100:
        raise ValueError("num_sites must be between 1 and 100")
    if num_orientations_per_site < 1 or num_orientations_per_site > 20:
        raise ValueError("num_orientations_per_site must be between 1 and 20")

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
    surface_normal = np.cross(slab_atoms.cell[0], slab_atoms.cell[1])
    surface_normal /= np.linalg.norm(surface_normal)
    heights = np.dot(slab_atoms.positions, surface_normal)
    tags = np.zeros(len(slab_atoms), dtype=int)
    tags[heights >= heights.max() - SURFACE_LAYER_TOLERANCE_A] = 1
    slab_atoms.set_tags(tags)

    np.random.seed(RANDOM_SEED)
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
        mode=PLACEMENT_MODE,
    )

    output_directory = slab_path.with_name(
        f"{slab_path.stem}_{adsorbate_path.stem}_candidates_"
        f"s{num_sites}_o{num_orientations_per_site}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    candidates = []
    for index, atoms in enumerate(configuration.atoms_list):
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
        metadata_path.write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "slab_atom_count": len(slab_atoms),
                    "adsorbate_atom_indices": adsorbate_indices,
                    "binding_atom_indices_within_adsorbate": binding_atom_indices,
                    "placement_metadata": _json_safe(placement_metadata),
                },
                separators=(",", ":"),
            )
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "structure_path": str(structure_path.resolve()),
                "preview_path": str(preview_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "num_atoms": len(atoms),
                "adsorbate_atom_indices": adsorbate_indices,
                "placement_metadata": _json_safe(placement_metadata),
            }
        )

    if not candidates:
        raise RuntimeError("FAIR Chemistry generated no adsorption candidates")

    return {
        "success": True,
        "error": None,
        "artifact_kind": "adsorption_candidate_set",
        "slab_structure_path": str(slab_path.resolve()),
        "adsorbate_structure_path": str(adsorbate_path.resolve()),
        "binding_atom_indices": binding_atom_indices,
        "binding_indexing": "zero_based",
        "candidate_count": len(candidates),
        "candidate_directory": str(output_directory.resolve()),
        "candidates": candidates,
        "generation_metadata": {
            "engine": "FAIR Chemistry AdsorbateSlabConfig",
            "placement_mode": PLACEMENT_MODE,
            "requested_num_sites": num_sites,
            "num_orientations_per_site": num_orientations_per_site,
            "random_seed": RANDOM_SEED,
            "surface_side": "top",
            "surface_atom_count": int(np.count_nonzero(tags == 1)),
            "relaxed": False,
            "energy_ranked": False,
        },
    }


def main() -> int:
    result = None
    try:
        if len(sys.argv) != 6:
            raise ValueError(
                "Usage: adsorption_candidate_runner.py SLAB ADSORBATE "
                "BINDING_INDICES NUM_SITES NUM_ORIENTATIONS"
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
            )
        exit_code = 0
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
        exit_code = 1

    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
