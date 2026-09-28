import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vaspilot.tools.mcp import mcp_server
from vaspilot.tools.mcp.bader_analysis import submit_bader_job
from vaspilot.tools.mcp.vasp_calculate import _read_calculation_result, check_status


@pytest.fixture
def completed_scf(tmp_path):
    scf = tmp_path / "scf"
    scf.mkdir()
    for name in ("CHGCAR", "AECCAR0", "AECCAR2"):
        (scf / name).write_text("charge data")
    return {"calc_type": "scf", "status": "completed", "calculate_path": str(scf)}


def test_submit_bader_uses_completed_scf_record_and_submits_slurm_job(
    completed_scf, tmp_path, monkeypatch
):
    binary = tmp_path / "bader"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    monkeypatch.setattr("vaspilot.tools.mcp.bader_analysis.subprocess.run", fake_run)
    result = submit_bader_job(
        "scf-1", completed_scf, tmp_path / "work", str(binary),
        python_executable="/python", partition="local"
    )

    assert result["success"] is True
    assert result["calc_type"] == "bader"
    assert result["restart_id"] == "scf-1"
    assert result["slurm_id"] == "12345"
    script = Path(result["calculate_path"]) / "bader.sbatch"
    assert script.is_file()
    assert "vaspilot.tools.mcp.bader_analysis" in script.read_text()
    assert calls[0][0][:3] == ["sbatch", "--parsable", "--partition=local"]


@pytest.mark.parametrize("change, expected", [
    ({"status": "running"}, "completed"),
    ({"calc_type": "relaxation"}, "SCF"),
])
def test_bader_rejects_wrong_source_record(completed_scf, tmp_path, change, expected):
    source = {**completed_scf, **change}
    result = submit_bader_job("scf-1", source, tmp_path, "bader")
    assert result["success"] is False
    assert result["retryable"] is False
    assert expected in result["error"]


def test_bader_rejects_missing_charge_files(completed_scf, tmp_path):
    (Path(completed_scf["calculate_path"]) / "AECCAR2").unlink()
    result = submit_bader_job("scf-1", completed_scf, tmp_path, "bader")
    assert result["success"] is False
    assert "AECCAR2" in result["error"]


def test_completed_bader_result_exposes_atom_electrons(tmp_path):
    output = tmp_path / "analysis"
    output.mkdir()
    result = {"success": True, "acf_path": str(output / "ACF.dat"),
              "atoms": [{"index": 1, "electrons": 0.75}]}
    (output / "ACF.dat").write_text("1 0 0 0 0.75 0.1 10\n")
    (output / "bader_result.json").write_text(json.dumps(result))
    parsed = _read_calculation_result("bader", str(output))
    assert parsed["status"] == "completed"
    assert parsed["bader_charges"] == [{"index": 1, "electrons": 0.75}]
    assert parsed["acf_path"] == str(output / "ACF.dat")



@pytest.mark.parametrize("has_result", [True, False])
def test_monitor_marks_bader_complete_only_with_result(tmp_path, monkeypatch, has_result):
    output = tmp_path / "bader"
    output.mkdir()
    if has_result:
        (output / "ACF.dat").write_text("1 0 0 0 0.75 0.1 10\n")
        (output / "bader_result.json").write_text(json.dumps({
            "success": True, "acf_path": str(output / "ACF.dat"),
            "atoms": [{"index": 1, "electrons": 0.75}],
        }))
    monkeypatch.setattr("vaspilot.tools.mcp.vasp_calculate.time.sleep", lambda _: None)

    def fake_run(args, **kwargs):
        if args[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="JobId=42 JobState=COMPLETED", stderr="")

    monkeypatch.setattr("vaspilot.tools.mcp.vasp_calculate.subprocess.run", fake_run)
    result = check_status({"bader-1": {
        "slurm_id": "42", "calc_type": "bader",
        "calculate_path": str(output), "status": "submitted", "success": True,
    }})["bader-1"]
    assert result["status"] == ("completed" if has_result else "failed")
    assert result["success"] is has_result
    if has_result:
        assert result["bader_charges"] == [{"index": 1, "electrons": 0.75}]

def test_mcp_tool_uses_scf_calculation_id(tmp_path, monkeypatch, completed_scf):
    registered = {}
    records = {"scf-1": completed_scf}

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
            pass

        def read_record(self, calculation_id):
            return records.get(calculation_id)

        def write_record(self, calculation_id, result):
            records[calculation_id] = result

    monkeypatch.setattr(mcp_server, "FastMCP", FakeMCP)
    monkeypatch.setattr(mcp_server, "VaspCalculationDB", FakeDB)
    real_submit = mcp_server.submit_bader_job
    monkeypatch.setattr(mcp_server, "submit_bader_job", lambda *args, **kwargs: {
        "success": True, "calculation_id": "bader-1", "slurm_id": "42",
        "status": "submitted", "calc_type": "bader",
        "calculate_path": str(tmp_path / "analysis"), "restart_id": "scf-1",
    })
    config = tmp_path / "mcp.yaml"
    config.write_text(json.dumps({
        "db_path": str(tmp_path / "db.sqlite"),
        "attachment_path": str(tmp_path),
        "mp_api_key": "unused", "structure_path": str(tmp_path),
        "work_dir": str(tmp_path), "bader_executable": "/bader",
    }))
    mcp_server.main(config_path=str(config))

    result = asyncio.run(registered["run_bader"](scf_calculation_id="scf-1"))
    assert result["success"] is True
    assert records["bader-1"]["restart_id"] == "scf-1"
    monkeypatch.setattr(mcp_server, "submit_bader_job", real_submit)
    assert asyncio.run(registered["run_bader"](scf_calculation_id="missing"))["success"] is False


def test_generated_bader_job_runs_with_a_space_in_source_path(tmp_path, monkeypatch):
    import subprocess
    import sys

    import numpy as np
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp import Chgcar, Poscar

    from vaspilot.tools.mcp import bader_analysis

    scf = tmp_path / "scf inputs"
    scf.mkdir()
    structure = Structure(Lattice.cubic(4), ["H"], [[0, 0, 0]])
    for name, value in (("CHGCAR", 3), ("AECCAR0", 1), ("AECCAR2", 2)):
        Chgcar(Poscar(structure), {"total": np.full((2, 2, 2), value)}).write_file(scf / name)
    binary = tmp_path / "fake bader"
    binary.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "Path('ACF.dat').write_text('# X Y Z CHARGE MIN DIST ATOMIC VOL\\n'"
        "'1 0 0 0 0.75 0.1 10\\n')\n"
    )
    binary.chmod(0o755)
    with monkeypatch.context() as patch:
        patch.setattr(bader_analysis.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="4242\n", stderr=""
        ))
        submitted = submit_bader_job(
            "scf-1", {"calc_type": "scf", "status": "completed", "calculate_path": str(scf)},
            tmp_path / "work", str(binary), python_executable=sys.executable
        )
    assert submitted["success"] is True
    output = Path(submitted["calculate_path"])
    process = subprocess.run(["bash", "bader.sbatch"], cwd=output, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert _read_calculation_result("bader", str(output))["bader_charges"] == [
        {"index": 1, "electrons": 0.75}
    ]


def test_bader_result_survives_database_round_trip(tmp_path):
    from vaspilot.tools.mcp.sqlite_database import VaspCalculationDB

    db = VaspCalculationDB(str(tmp_path / "record.db"))
    expected = [{"index": 1, "electrons": 0.75}]
    db.write_record("bader-1", {
        "calc_type": "bader", "status": "completed", "success": True,
        "calculate_path": str(tmp_path), "restart_id": "scf-1",
        "slurm_id": "42", "acf_path": str(tmp_path / "ACF.dat"),
        "bader_charges": expected,
    })
    restored = VaspCalculationDB(str(tmp_path / "record.db")).read_record("bader-1")
    assert restored["acf_path"] == str(tmp_path / "ACF.dat")
    assert restored["bader_charges"] == expected


def test_failed_bader_job_reports_its_stderr(tmp_path, monkeypatch):
    output = tmp_path / "bader"
    output.mkdir()
    (output / "bader.err").write_text("Bader binary crashed")
    monkeypatch.setattr("vaspilot.tools.mcp.vasp_calculate.time.sleep", lambda _: None)

    def fake_run(args, **kwargs):
        if args[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="JobId=42 JobState=FAILED", stderr="")

    monkeypatch.setattr("vaspilot.tools.mcp.vasp_calculate.subprocess.run", fake_run)
    result = check_status({"bader-1": {
        "slurm_id": "42", "calc_type": "bader", "calculate_path": str(output),
        "status": "submitted", "success": True,
    }})["bader-1"]
    assert result["status"] == "failed"
    assert "Bader binary crashed" in result["error"]
    assert result["success"] is False


def test_existing_database_gets_bader_columns(tmp_path):
    import sqlite3

    from vaspilot.tools.mcp.sqlite_database import VaspCalculationDB

    path = tmp_path / "old.db"
    VaspCalculationDB(str(path))
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE calculations DROP COLUMN acf_path")
        connection.execute("ALTER TABLE calculations DROP COLUMN bader_charges_blob")
    db = VaspCalculationDB(str(path))
    db.write_record("bader-1", {"calc_type": "bader", "status": "completed",
                                "acf_path": "ACF.dat",
                                "bader_charges": [{"index": 1, "electrons": 0.75}]})
    assert db.read_record("bader-1")["bader_charges"] == [{"index": 1, "electrons": 0.75}]


def test_monitor_does_not_mark_unreadable_scf_complete(tmp_path, monkeypatch):
    monkeypatch.setattr("vaspilot.tools.mcp.vasp_calculate.time.sleep", lambda _: None)
    monkeypatch.setattr("vaspilot.tools.mcp.vasp_calculate._read_calculation_result",
                        lambda *args: {"success": False, "error": "SCF output unreadable"})

    def fake_run(args, **kwargs):
        if args[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="JobId=42 JobState=COMPLETED", stderr="")

    monkeypatch.setattr("vaspilot.tools.mcp.vasp_calculate.subprocess.run", fake_run)
    result = check_status({"scf-1": {
        "slurm_id": "42", "calc_type": "scf",
        "calculate_path": str(tmp_path), "status": "submitted", "success": True,
    }})["scf-1"]
    assert result["status"] == "failed"
    assert result["success"] is False


@pytest.mark.parametrize("calc_type", ["bader", "relaxation", "scf"])
def test_completed_calculation_is_read_from_database_without_rechecking_slurm(
    tmp_path, monkeypatch, calc_type
):
    registered = {}
    saved = {"calc_type": calc_type, "status": "completed", "success": True,
             "slurm_id": "42", "calculate_path": str(tmp_path),
             "acf_path": str(tmp_path / "ACF.dat"),
             "bader_charges": [{"index": 1, "electrons": 0.75}]}

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
            pass
        def read_record(self, calculation_id):
            return saved if calculation_id == "old-job" else None
        def write_record(self, calculation_id, data):
            raise AssertionError("Completed calculation should not be rewritten")

    monkeypatch.setattr(mcp_server, "FastMCP", FakeMCP)
    monkeypatch.setattr(mcp_server, "VaspCalculationDB", FakeDB)
    monkeypatch.setattr(mcp_server, "check_status", lambda records: (_ for _ in ()).throw(
        AssertionError("Finished calculation should not query SLURM")
    ))
    config = tmp_path / "mcp.yaml"
    config.write_text(json.dumps({
        "db_path": str(tmp_path / "db.sqlite"), "attachment_path": str(tmp_path),
        "mp_api_key": "unused", "structure_path": str(tmp_path),
        "work_dir": str(tmp_path),
    }))
    mcp_server.main(config_path=str(config))
    result = asyncio.run(registered["check_calculation_status"](["old-job"]))
    assert result["old-job"]["status"] == "completed"
    if calc_type == "bader":
        assert result["old-job"]["bader_charges"] == saved["bader_charges"]
