from typing import Any, Dict, List, Optional

from .adsorption_tools import (
    generate_adsorption_candidates,
    relax_adsorption_candidates,
)
from .struct_tools import build_adsorbate, build_surface, retrieve_bulk_parent


def run_adsorption_workflow(
    *,
    api_key: str,
    download_path: str,
    miller_index: List[int],
    adsorbate: str,
    formula: Optional[str] = None,
    material_id: Optional[str] = None,
    num_sites: int = 10,
    num_orientations_per_site: int = 1,
    fmax: float = 0.05,
    max_steps: int = 200,
) -> Dict[str, Any]:
    """Run the artifact-producing adsorption workflow without LLM hand-offs."""
    stages: Dict[str, Any] = {}

    def fail(stage: str, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": False,
            "failed_stage": stage,
            "error": result.get("error") or f"{stage} failed",
            "stages": stages,
        }

    bulk = retrieve_bulk_parent(
        api_key=api_key,
        download_path=download_path,
        formula=formula,
        material_id=material_id,
    )
    stages["bulk_retrieval"] = bulk
    if not bulk.get("success"):
        return fail("bulk_retrieval", bulk)

    slab = build_surface(bulk["structure_path"], miller_index)
    stages["surface_construction"] = slab
    if not slab.get("success"):
        return fail("surface_construction", slab)

    molecule = build_adsorbate(adsorbate, download_path)
    stages["adsorbate_construction"] = molecule
    if not molecule.get("success"):
        return fail("adsorbate_construction", molecule)

    candidates = generate_adsorption_candidates(
        slab_structure_path=slab["slab_structure_path"],
        adsorbate_structure_path=molecule["adsorbate_structure_path"],
        binding_atom_indices=molecule["binding_atom_indices"],
        num_sites=num_sites,
        num_orientations_per_site=num_orientations_per_site,
    )
    stages["candidate_generation"] = candidates
    if not candidates.get("success"):
        return fail("candidate_generation", candidates)

    expected_atoms = int(slab["num_atoms"]) + int(molecule["num_atoms"])
    invalid = [
        item.get("structure_path", item.get("candidate_id", "unknown"))
        for item in candidates.get("candidates", [])
        if item.get("num_atoms") != expected_atoms
    ]
    if invalid:
        return fail(
            "candidate_validation",
            {"error": f"Candidates have an unexpected atom count: {invalid}"},
        )

    candidate_paths = [
        item["structure_path"] for item in candidates.get("candidates", [])
    ]
    if not candidate_paths:
        return fail("candidate_validation", {"error": "No candidate paths returned"})

    relaxation = relax_adsorption_candidates(
        candidate_structure_paths=candidate_paths,
        fmax=fmax,
        max_steps=max_steps,
    )
    stages["candidate_relaxation"] = relaxation
    if not relaxation.get("success"):
        return fail("candidate_relaxation", relaxation)

    return {
        "success": True,
        "error": None,
        "workflow_kind": "adsorption_screening",
        "expected_atoms_per_candidate": expected_atoms,
        "candidate_structure_paths": candidate_paths,
        "stages": stages,
        "relaxation": relaxation,
    }
