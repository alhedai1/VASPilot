import json
from types import SimpleNamespace

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
