import json
from pathlib import Path

import pytest

from vaspilot.tools.mcp.adsorption_analysis import (
    load_analysis_scf_submission,
    prepare_adsorption_analysis_scf,
    write_analysis_scf_submission,
)


def _manifest(tmp_path):
    paths = []
    for name in ("candidate_000", "candidate_001"):
        path = tmp_path / f"{name}.vasp"
        path.write_text(f"structure-{name}")
        paths.append(path)
    return {
        "artifact_kind": "ranked_adsorption_relaxations",
        "artifact_id": "relax-123",
        "results": [
            {
                "candidate_id": "candidate_000",
                "candidate_index": 0,
                "rank": 2,
                "status": "converged",
                "final_energy_eV": -9.0,
                "final_structure_path": str(paths[0]),
                "geometry_diagnostics": {"desorbed": False},
            },
            {
                "candidate_id": "candidate_001",
                "candidate_index": 1,
                "rank": 1,
                "status": "converged",
                "final_energy_eV": -10.0,
                "final_structure_path": str(paths[1]),
                "geometry_diagnostics": {"desorbed": False},
            },
        ],
    }


def test_prepares_best_candidate_static_scf_artifact(tmp_path):
    result = prepare_adsorption_analysis_scf(
        _manifest(tmp_path), str(tmp_path / "outputs")
    )

    assert result["selected_candidate_id"] == "candidate_001"
    assert result["selection_policy"] == "lowest_final_uma_energy"
    assert result["submitted"] is False
    assert result["vasp_scf_arguments"]["soc"] is False
    assert result["vasp_scf_arguments"]["incar_tags"]["NSW"] == 0
    assert result["vasp_scf_arguments"]["incar_tags"]["LAECHG"] is True
    assert result["required_outputs"]["bader"] == ["CHGCAR", "AECCAR0", "AECCAR2"]
    assert result["required_outputs"]["lobster"] == ["WAVECAR", "POSCAR", "POTCAR"]
    assert json.loads(open(result["manifest_path"]).read())["artifact_id"] == result["artifact_id"]


def test_explicit_candidate_selection_and_deterministic_artifact(tmp_path):
    kwargs = dict(
        relaxation_manifest=_manifest(tmp_path),
        output_directory=str(tmp_path / "outputs"),
        candidate_id="candidate_000",
        analyses=["bader"],
        kpoint_num=[3, 3, 1],
    )
    first = prepare_adsorption_analysis_scf(**kwargs)
    second = prepare_adsorption_analysis_scf(**kwargs)
    assert first["artifact_id"] == second["artifact_id"]
    assert first["selected_candidate_id"] == "candidate_000"
    assert first["required_outputs"]["lobster"] == []


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"analyses": ["dos"]}, "analyses"),
        ({"analyses": []}, "analyses"),
        ({"kpoint_num": [4, 4, 0]}, "kpoint_num"),
        ({"incar_overrides": {"NSW": 1}}, "NSW=0"),
        ({"incar_overrides": {"LAECHG": False}}, "LAECHG=true"),
        ({"incar_overrides": {"LSORBIT": True}}, "SOC is unsupported"),
    ],
)
def test_rejects_incompatible_analysis_scf_settings(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        prepare_adsorption_analysis_scf(
            _manifest(tmp_path), str(tmp_path / "outputs"), **kwargs
        )


def test_rejects_failed_or_geometrically_invalid_candidates(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["results"][0]["status"] = "not_converged"
    manifest["results"][1]["geometry_diagnostics"]["desorbed"] = True
    with pytest.raises(ValueError, match="No converged"):
        prepare_adsorption_analysis_scf(manifest, str(tmp_path / "outputs"))


def test_loads_exact_prepared_scf_arguments(tmp_path):
    prepared = prepare_adsorption_analysis_scf(
        _manifest(tmp_path), str(tmp_path / "outputs")
    )
    path, arguments = load_analysis_scf_submission(prepared["manifest_path"])

    assert path == Path(prepared["manifest_path"])
    assert arguments == prepared["vasp_scf_arguments"]


def test_rejects_modified_prepared_structure(tmp_path):
    prepared = prepare_adsorption_analysis_scf(
        _manifest(tmp_path), str(tmp_path / "outputs")
    )
    Path(prepared["structure_path"]).write_text("modified")

    with pytest.raises(ValueError, match="provenance hash"):
        load_analysis_scf_submission(prepared["manifest_path"])


def test_records_submission_and_prevents_resubmission(tmp_path):
    prepared = prepare_adsorption_analysis_scf(
        _manifest(tmp_path), str(tmp_path / "outputs")
    )
    path, _ = load_analysis_scf_submission(prepared["manifest_path"])
    record = write_analysis_scf_submission(
        path,
        {
            "success": True,
            "calculation_id": "calc-1",
            "slurm_id": "42",
            "calculate_path": "/work/calc-1/scf",
            "status": "pending",
            "error": None,
        },
    )

    assert record["submitted"] is True
    assert record["calculation_id"] == "calc-1"
    assert json.loads(Path(record["submission_manifest_path"]).read_text()) == record
    with pytest.raises(ValueError, match="already been submitted"):
        load_analysis_scf_submission(prepared["manifest_path"])
