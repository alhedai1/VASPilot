import json

import numpy as np
import pytest
from ase import Atoms

from vaspilot.scripts.adsorption_relaxation_runner import (
    geometry_diagnostics,
    rank_converged,
    validate_request,
)


def test_accepts_typed_candidate_set_and_preserves_set_id():
    config = validate_request({
        "candidate_set": {
            "artifact_id": "set-123",
            "candidates": [{"structure_path": "candidate_000.vasp"}],
        }
    })
    assert config["candidate_structure_paths"] == ["candidate_000.vasp"]
    assert config["candidate_set_id"] == "set-123"


def test_accepts_persisted_candidate_manifest(tmp_path):
    manifest = tmp_path / "candidate_manifest.json"
    manifest.write_text(json.dumps({
        "artifact_id": "set-456",
        "candidates": [{"structure_path": "candidate_001.vasp"}],
    }))
    config = validate_request({"candidate_manifest_path": str(manifest)})
    assert config["candidate_structure_paths"] == ["candidate_001.vasp"]
    assert config["candidate_set_id"] == "set-456"


@pytest.mark.parametrize("key,value", [
    ("device", "tpu"), ("precision", "float16"),
    ("optimizer", "Unknown"), ("max_steps", 0),
])
def test_rejects_invalid_relaxation_configuration(key, value):
    with pytest.raises(ValueError):
        validate_request({"candidate_structure_paths": ["a.vasp"], key: value})


def test_ranks_only_converged_candidates_deterministically():
    results = [
        {"candidate_index": 0, "status": "not_converged", "final_energy_eV": -10.0, "rank": None},
        {"candidate_index": 1, "status": "converged", "final_energy_eV": -8.0, "rank": None},
        {"candidate_index": 2, "status": "converged", "final_energy_eV": -9.0, "rank": None},
    ]
    ranked = rank_converged(results)
    assert [item["candidate_index"] for item in ranked] == [2, 1]
    assert [item["rank"] for item in ranked] == [1, 2]
    assert results[0]["rank"] is None
    assert ranked[1]["energy_relative_to_best_eV"] == pytest.approx(1.0)


def test_reports_desorption_fragmentation_penetration_and_reconstruction():
    initial = Atoms(
        "Pt2CO", positions=[[0, 0, 0], [2, 0, 0], [0, 0, 2], [0, 0, 3.1]],
        cell=[10, 10, 20], pbc=True,
    )
    final = initial.copy()
    final.positions[1] += [3, 0, 0]
    final.positions[2] = [0, 0, -2]
    final.positions[3] = [0, 0, 7]
    config = validate_request({"candidate_structure_paths": ["a.vasp"]})
    diagnostics = geometry_diagnostics(initial, final, [2, 3], config)
    assert diagnostics["fragmented"] is True
    assert diagnostics["penetrated_surface"] is True
    assert diagnostics["severe_slab_reconstruction"] is True
    assert isinstance(diagnostics["desorbed"], bool)
