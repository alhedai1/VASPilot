import json
from types import SimpleNamespace

from vaspilot.tools.mcp import adsorption_tools


def test_generate_adsorption_candidates_calls_external_runner(monkeypatch):
    expected = {"success": True, "candidate_count": 4, "candidates": []}

    def fake_run(command, capture_output, text):
        assert command[2:5] == ["slab.vasp", "CO.xyz", "1"]
        assert command[5:] == ["10", "2"]
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
        assert json.loads(command[2]) == ["candidate_000.vasp", "candidate_001.vasp"]
        assert command[3:] == ["0.05", "200"]
        return SimpleNamespace(stdout=json.dumps(expected), stderr="", returncode=0)

    monkeypatch.setattr(adsorption_tools.subprocess, "run", fake_run)

    result = adsorption_tools.relax_adsorption_candidates(
        ["candidate_000.vasp", "candidate_001.vasp"]
    )

    assert result == expected
