from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Kpoints

from vaspilot.tools.mcp import vasp_calculate
from vaspilot.tools.mcp.sqlite_database import VaspCalculationDB


def test_scf_copies_source_potcar_and_records_actual_inputs(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    source.mkdir()
    potcar = source / "POTCAR"
    potcar.write_bytes(b"   TITEL  = PAW_PBE Pd_pv 06Sep2000\n   ENMAX  = 271.098; ENMIN = 203.323 eV\n")
    monkeypatch.setattr(vasp_calculate, "_submit_slurm_job", lambda kind, path, attachment: {
        "success": True, "calc_type": kind, "status": "submitted", "slurm_id": "42",
        "calculate_path": path, "error": None,
    })

    result = vasp_calculate.vasp_scf(
        calculation_id="scf-1", work_dir=str(tmp_path),
        struct=Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
        kpoints=Kpoints.gamma_automatic(kpts=(12, 12, 1)),
        incar_dict={"ENCUT": 300, "NSW": 0, "IBRION": -1, "ICHARG": 2},
        source_potcar_path=str(potcar),
    )

    scf_dir = tmp_path / "scf-1" / "scf"
    assert (scf_dir / "POTCAR").read_bytes() == potcar.read_bytes()
    assert result["input_provenance"]["potcar_titles"] == ["PAW_PBE Pd_pv 06Sep2000"]
    assert result["input_provenance"]["encut_eV"] == 300.0
    assert result["input_provenance"]["kpoint_mesh"] == [12, 12, 1]


def test_database_persists_input_provenance(tmp_path):
    db = VaspCalculationDB(str(tmp_path / "calculations.db"))
    provenance = {"potcar_titles": ["PAW_PBE Pd_pv 06Sep2000"],
                  "potcar_sha256": "abc", "encut_eV": 300.0,
                  "kpoint_mesh": [12, 12, 1]}
    db.write_record("relax-1", {
        "success": True, "calc_type": "relaxation", "status": "submitted",
        "input_provenance": provenance,
    })
    assert db.read_record("relax-1")["input_provenance"] == provenance


def test_attachment_cannot_replace_generated_vasp_inputs(tmp_path, monkeypatch):
    calculation = tmp_path / "calculation"
    attachment = tmp_path / "attachment"
    calculation.mkdir()
    attachment.mkdir()
    (calculation / "POTCAR").write_text("chosen potential")
    (attachment / "POTCAR").write_text("different potential")
    (attachment / "slurm.sh").write_text("#!/bin/sh\n")
    monkeypatch.setattr(vasp_calculate.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("job submitted")))

    result = vasp_calculate._submit_slurm_job("scf", str(calculation), str(attachment))

    assert result["success"] is False
    assert "POTCAR" in result["error"]
    assert (calculation / "POTCAR").read_text() == "chosen potential"


def test_attachment_cannot_supply_restart_file_implicitly(tmp_path, monkeypatch):
    calculation = tmp_path / "calculation"
    attachment = tmp_path / "attachment"
    calculation.mkdir()
    attachment.mkdir()
    (attachment / "WAVECAR").write_text("old wavefunctions")
    (attachment / "slurm.sh").write_text("#!/bin/sh\n")
    monkeypatch.setattr(vasp_calculate.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("job submitted")))

    result = vasp_calculate._submit_slurm_job("scf", str(calculation), str(attachment))

    assert result["success"] is False
    assert "WAVECAR" in result["error"]
    assert not (calculation / "WAVECAR").exists()


def test_provenance_summarizes_large_magnetic_moment_list(tmp_path):
    from vaspilot.tools.mcp.vasp_input_consistency import input_provenance

    (tmp_path / "POTCAR").write_text("   TITEL  = PAW_PBE Pd 05Jan2001\n   ENMAX  = 250.925; ENMIN = 188.194 eV\n")
    (tmp_path / "KPOINTS").write_text("Automatic kpoint scheme\n0\nGamma\n4 4 1\n")
    (tmp_path / "INCAR").write_text("MAGMOM = 1000*0.0\nISPIN = 2\n")

    provenance = input_provenance(tmp_path)

    assert provenance["electronic_settings"]["MAGMOM"]["count"] == 1000
    assert len(str(provenance)) < 1000


def test_status_error_preserves_input_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(vasp_calculate.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(vasp_calculate.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("scheduler unavailable")))
    record = {"slurm_id": "42", "calc_type": "scf", "calculate_path": str(tmp_path),
              "status": "submitted", "input_provenance": {"potcar_sha256": "abc"}}

    result = vasp_calculate.check_status({"scf-1": record})

    assert result["scf-1"]["status"] == "error"
    assert result["scf-1"]["input_provenance"] == {"potcar_sha256": "abc"}


def test_expired_slurm_job_recovers_completed_vasp_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "OUTCAR").write_text(
        "General timing and accounting informations for this job:\n")
    (tmp_path / "vasprun.xml").write_text("complete VASP output")
    monkeypatch.setattr(vasp_calculate.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(vasp_calculate.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="",
                                        stderr="slurm_load_jobs error: Invalid job id specified"))
    monkeypatch.setattr(vasp_calculate, "Vasprun", lambda *args, **kwargs:
                        SimpleNamespace(converged=True))
    monkeypatch.setattr(vasp_calculate, "_read_calculation_result", lambda *args: {
        "total_energy": -187.25900531, "status": "completed"})
    record = {"slurm_id": "95", "calc_type": "relaxation",
              "calculate_path": str(tmp_path), "status": "unknown",
              "error": "old scheduler error", "success": False}

    result = vasp_calculate.check_status({"relax-1": record})["relax-1"]

    assert result["status"] == "completed"
    assert result["success"] is True
    assert result["error"] is None
    assert result["total_energy"] == -187.25900531


def test_expired_slurm_job_without_complete_output_stays_unknown(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "vasprun.xml").write_text("partial output")
    monkeypatch.setattr(vasp_calculate.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(vasp_calculate.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="",
                                        stderr="slurm_load_jobs error: Invalid job id specified"))
    record = {"slurm_id": "95", "calc_type": "relaxation",
              "calculate_path": str(tmp_path), "status": "submitted"}

    result = vasp_calculate.check_status({"relax-1": record})["relax-1"]

    assert result["status"] == "unknown"
    assert result["success"] is False


def test_expired_slurm_job_with_unparseable_output_stays_unknown(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "OUTCAR").write_text(
        "General timing and accounting informations for this job:\n")
    (tmp_path / "vasprun.xml").write_text("corrupt XML")
    monkeypatch.setattr(vasp_calculate.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(vasp_calculate.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="",
                                        stderr="slurm_load_jobs error: Invalid job id specified"))
    monkeypatch.setattr(vasp_calculate, "Vasprun", lambda *args, **kwargs:
                        (_ for _ in ()).throw(RuntimeError("corrupt XML")))
    record = {"slurm_id": "95", "calc_type": "relaxation",
              "calculate_path": str(tmp_path), "status": "submitted"}

    result = vasp_calculate.check_status({"relax-1": record})["relax-1"]

    assert result["status"] == "unknown"
    assert result["success"] is False
