import asyncio
import json

import pytest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from vaspilot.tools.mcp import mcp_server


def _registered_tools(tmp_path, monkeypatch):
    registered = {}

    class FakeMCP:
        def __init__(self, name):
            pass

        def tool(self, *, name):
            def register(function):
                registered[name] = function
                return function

            return register

        def run(self, **kwargs):
            pass

    class FakeDB:
        def __init__(self, db_path):
            self.records = {}

        def write_record(self, calculation_id, result):
            self.records[calculation_id] = result

        def read_record(self, calculation_id):
            return self.records.get(calculation_id)

    monkeypatch.setattr(mcp_server, "FastMCP", FakeMCP)
    monkeypatch.setattr(mcp_server, "VaspCalculationDB", FakeDB)
    config = tmp_path / "mcp.yaml"
    config.write_text(json.dumps({
        "db_path": str(tmp_path / "db.sqlite"),
        "attachment_path": str(tmp_path),
        "mp_api_key": "unused",
        "structure_path": str(tmp_path),
        "work_dir": str(tmp_path),
        "VASP_default_INCAR": {"relaxation": {"NSW": 100}},
    }))
    mcp_server.main(config_path=str(config))
    return registered



@pytest.fixture
def relaxation(tmp_path, monkeypatch):
    tool = _registered_tools(tmp_path, monkeypatch)["vasp_relaxation"]
    structure = tmp_path / "arbitrary-output.vasp"
    Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]).to(str(structure), fmt="poscar")
    manifest = tmp_path / "relaxation_manifest.json"
    data = {"artifact_kind": "ranked_adsorption_relaxations", "success": True,
            "results": [{"candidate_id": "candidate_002", "converged": True,
                         "final_structure_path": structure.name}]}
    manifest.write_text(json.dumps(data))
    submissions = []
    def submit(**kwargs):
        submissions.append(kwargs)
        return {"success": True, "slurm_id": "42", "status": "submitted",
                "error": None, "calculate_path": str(tmp_path / "job")}
    monkeypatch.setattr(mcp_server, "vasp_relaxation", submit)
    return tool, manifest, data, structure, submissions


@pytest.mark.parametrize("absolute", [False, True])
def test_manifest_selects_exact_candidate_structure(relaxation, absolute):
    tool, manifest, data, structure, submissions = relaxation
    if absolute:
        data["results"][0]["final_structure_path"] = str(structure)
    data["results"].insert(0, {"candidate_id": "candidate_001", "converged": True,
                               "final_structure_path": "not-selected.vasp"})
    manifest.write_text(json.dumps(data))
    result = asyncio.run(tool(relaxation_manifest_path=str(manifest),
                              candidate_id="candidate_002", kpoint_num=(3, 3, 1),
                              incar_tags={"NSW": 50}))
    assert result["success"] is True
    assert result["slurm_id"] == "42"
    assert len(submissions) == 1
    assert submissions[0]["struct"].composition.reduced_formula == "Pd"
    assert submissions[0]["incar_dict"]["NSW"] == 50
    assert submissions[0]["kpoints"].kpts == [(3, 3, 1)]


def test_direct_structure_path_remains_supported(relaxation):
    tool, _, _, structure, submissions = relaxation
    result = asyncio.run(tool(structure_path=str(structure)))
    assert result["success"] is True
    assert len(submissions) == 1


@pytest.mark.parametrize("case, code", [
    ("missing_manifest", "INVALID_RELAXATION_INPUT"),
    ("invalid_json", "INVALID_RELAXATION_INPUT"),
    ("wrong_kind", "INVALID_RELAXATION_INPUT"),
    ("bad_results", "INVALID_RELAXATION_INPUT"),
    ("unknown_candidate", "INVALID_RELAXATION_INPUT"),
    ("duplicate_candidate", "INVALID_RELAXATION_INPUT"),
    ("unconverged", "INVALID_RELAXATION_INPUT"),
    ("missing_structure", "INVALID_RELAXATION_INPUT"),
    ("unreadable_structure", "INVALID_RELAXATION_INPUT"),
    ("conflicting_inputs", "INVALID_RELAXATION_INPUT"),
    ("missing_candidate", "INVALID_RELAXATION_INPUT"),
    ("candidate_without_manifest", "INVALID_RELAXATION_INPUT"),
    ("no_inputs", "INVALID_RELAXATION_INPUT"),
    ("missing_direct_structure", "INVALID_RELAXATION_INPUT"),
])
def test_invalid_inputs_stop_before_submission(relaxation, case, code):
    tool, manifest, data, structure, submissions = relaxation
    kwargs = {"relaxation_manifest_path": str(manifest), "candidate_id": "candidate_002"}
    if case == "wrong_kind": data["artifact_kind"] = "adsorption_candidates"
    if case == "bad_results": data["results"] = [None]
    if case == "unknown_candidate": kwargs["candidate_id"] = "invented"
    if case == "duplicate_candidate": data["results"] *= 2
    if case == "unconverged": data["results"][0]["converged"] = False
    if case == "missing_structure": structure.unlink()
    if case == "unreadable_structure": structure.write_text("not a structure")
    if case == "conflicting_inputs": kwargs["structure_path"] = str(structure)
    if case == "missing_candidate": kwargs.pop("candidate_id")
    if case == "candidate_without_manifest": kwargs.pop("relaxation_manifest_path")
    if case == "no_inputs": kwargs = {}
    if case == "missing_direct_structure": kwargs = {"structure_path": str(structure)+"missing"}
    manifest.write_text(json.dumps(data))
    if case == "missing_manifest": manifest.unlink()
    if case == "invalid_json": manifest.write_text("{")
    result = asyncio.run(tool(**kwargs))
    assert result["success"] is False
    assert result["retryable"] is False
    assert result["error_code"] == code
    assert result.get("calculation_id") is None
    assert result.get("slurm_id") is None
    assert submissions == []


def test_relaxation_accepts_output_directory_reference(relaxation):
    tool, manifest, _, _, submissions = relaxation
    result = asyncio.run(tool(relaxation_manifest_path=str(manifest.parent),
                              candidate_id="candidate_002"))
    assert result["success"] is True
    assert len(submissions) == 1


def test_uma_relaxation_response_leads_with_manifest_path(tmp_path, monkeypatch):
    tools = _registered_tools(tmp_path, monkeypatch)
    manifest = str(tmp_path / "relaxation_manifest.json")
    monkeypatch.setattr(mcp_server, "relax_adsorption_candidates", lambda **kwargs: {
        "success": True,
        "output_directory": str(tmp_path),
        "provenance": {"long_diagnostics": "x" * 2000},
        "manifest_path": manifest,
        "best_candidate_id": "candidate_002",
    })
    result = asyncio.run(tools["relax_adsorption_candidates"]())
    assert '"manifest_path"' in json.dumps(result)[:100]
    assert result["manifest_path"] == manifest
