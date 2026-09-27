import json
from types import SimpleNamespace

import pytest

from vaspilot.scripts.adsorption_relaxation_runner import validate_request
from vaspilot.tools.mcp import adsorption_tools


def test_generate_adsorption_candidates_calls_external_runner(monkeypatch):
    expected = {"success": True, "candidate_count": 4, "candidates": []}

    def fake_run(command, capture_output, text):
        assert command[2:5] == ["slab.vasp", "CO.xyz", "1"]
        assert command[5:] == [
            "10", "2", "1000", "heuristic", "0", "0.1", "0.5"
        ]
        assert capture_output is True
        assert text is True
        return SimpleNamespace(
            stdout=json.dumps(expected), stderr="", returncode=0
        )

    monkeypatch.setattr(adsorption_tools.subprocess, "run", fake_run)

    result = adsorption_tools.generate_adsorption_candidates(
        "slab.vasp", "CO.xyz", [1], num_sites=10, num_orientations_per_site=2
    )

    assert result == expected


def test_generate_adsorption_candidates_forwards_general_parameters(monkeypatch):
    def fake_run(command, **kwargs):
        assert command[5:] == [
            "7", "3", "1000", "random_site_heuristic_placement", "42", "0.2", "0.3"
        ]
        return SimpleNamespace(stdout='{"success":true}', stderr="", returncode=0)

    monkeypatch.setattr(adsorption_tools.subprocess, "run", fake_run)
    result = adsorption_tools.generate_adsorption_candidates(
        "slab.vasp",
        "CO.xyz",
        [1],
        num_sites=7,
        num_orientations_per_site=3,
        placement_mode="random_site_heuristic_placement",
        random_seed=42,
        interstitial_gap=0.2,
        surface_layer_tolerance=0.3,
    )
    assert result["success"] is True


def test_generate_adsorption_candidates_handles_invalid_json(monkeypatch):
    monkeypatch.setattr(
        adsorption_tools.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="not-json", stderr="runner failed", returncode=1
        ),
    )

    result = adsorption_tools.generate_adsorption_candidates(
        "slab.vasp", "CO.xyz", [1]
    )

    assert result["success"] is False
    assert "runner returned invalid JSON" in result["error"]
    assert "runner failed" in result["error"]


@pytest.mark.parametrize("candidate_count", [4, 40])
def test_compact_candidates_preserve_manifest_handoff(
    tmp_path, monkeypatch, candidate_count
):
    manifest = tmp_path / "candidate_manifest.json"
    candidates = [
        {
            "candidate_id": f"candidate_{index:03d}",
            "structure_path": str(tmp_path / f"candidate_{index:03d}.vasp"),
            "geometry_diagnostics": {"all_adsorbate_atoms_above_top_surface": True},
        }
        for index in range(candidate_count)
    ]
    full_result = {
        "success": True,
        "error": None,
        "artifact_kind": "adsorption_candidate_set",
        "artifact_id": "set-1",
        "candidate_count": candidate_count,
        "manifest_path": str(manifest),
        "candidates": candidates,
        "generation_metadata": {
            "requested_candidate_upper_bound": 40,
            "available_site_count": candidate_count,
            "candidate_count_semantics": "up_to_requested_sites_times_orientations",
        },
    }
    manifest.write_text(json.dumps(full_result))
    original_manifest = manifest.read_bytes()
    monkeypatch.setattr(
        adsorption_tools.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(full_result), stderr="", returncode=0
        ),
    )

    result = adsorption_tools.generate_adsorption_candidates(
        "slab.vasp", "CO.xyz", [1], num_sites=40,
        include_candidate_details=False,
    )

    assert result["success"] is True
    assert result["error"] is None
    assert result["candidate_count"] == candidate_count
    assert result["generation_metadata"] == full_result["generation_metadata"]
    assert "candidates" not in result
    assert len(json.dumps(result)) < 2000
    assert manifest.read_bytes() == original_manifest
    config = validate_request({"candidate_manifest_path": result["manifest_path"]})
    assert config["candidate_set_id"] == "set-1"
    assert config["candidate_structure_paths"] == [
        str(tmp_path / f"candidate_{index:03d}.vasp")
        for index in range(candidate_count)
    ]


def test_compact_candidates_preserve_generation_errors(monkeypatch):
    error = {
        "success": False,
        "error_code": "candidate_generation_failed",
        "error": "FAIR Chemistry generated no adsorption candidates",
        "retryable": False,
    }
    monkeypatch.setattr(
        adsorption_tools.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(error), stderr="", returncode=1
        ),
    )
    result = adsorption_tools.generate_adsorption_candidates(
        "slab.vasp", "CO.xyz", [1], include_candidate_details=False,
    )
    assert result == error


def test_relax_adsorption_candidates_calls_external_runner(monkeypatch):
    expected = {"success": True, "best_candidate_id": "candidate_001"}

    def fake_run(command, capture_output, text):
        request = json.loads(command[2])
        assert request["candidate_structure_paths"] == [
            "candidate_000.vasp", "candidate_001.vasp"
        ]
        assert request["candidate_set"] is None
        assert request["candidate_manifest_path"] is None
        assert request["fmax"] == 0.05
        assert request["max_steps"] == 200
        assert request["model"] == "uma-s-1p2"
        assert request["optimizer"] == "LBFGS"
        assert len(command) == 3
        return SimpleNamespace(stdout=json.dumps(expected), stderr="", returncode=0)

    monkeypatch.setattr(adsorption_tools.subprocess, "run", fake_run)

    result = adsorption_tools.relax_adsorption_candidates(
        ["candidate_000.vasp", "candidate_001.vasp"]
    )

    assert result == expected


def test_relax_adsorption_candidates_forwards_typed_set_and_configuration(monkeypatch):
    candidate_set = {
        "artifact_id": "set-1",
        "candidates": [{"structure_path": "candidate_000.vasp"}],
    }

    def fake_run(command, **kwargs):
        request = json.loads(command[2])
        assert request["candidate_structure_paths"] is None
        assert request["candidate_set"] == candidate_set
        assert request["device"] == "cpu"
        assert request["precision"] == "float64"
        assert request["optimizer"] == "FIRE"
        return SimpleNamespace(stdout='{"success":true}', stderr="", returncode=0)

    monkeypatch.setattr(adsorption_tools.subprocess, "run", fake_run)
    result = adsorption_tools.relax_adsorption_candidates(
        candidate_set=candidate_set,
        device="cpu",
        precision="float64",
        optimizer="FIRE",
    )
    assert result["success"] is True


def test_relax_adsorption_candidates_forwards_manifest_path(monkeypatch):
    def fake_run(command, **kwargs):
        request = json.loads(command[2])
        assert request["candidate_manifest_path"] == "/tmp/candidate_manifest.json"
        assert request["candidate_set"] is None
        return SimpleNamespace(stdout='{"success":true}', stderr="", returncode=0)

    monkeypatch.setattr(adsorption_tools.subprocess, "run", fake_run)
    result = adsorption_tools.relax_adsorption_candidates(
        candidate_manifest_path="/tmp/candidate_manifest.json"
    )
    assert result["success"] is True
