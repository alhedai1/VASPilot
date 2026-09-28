import asyncio
import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from vaspilot.tools.mcp import mcp_server


def _registered_tools(tmp_path, monkeypatch, records=None, scf_defaults=None):
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
            self.records = records if records is not None else {}

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
        "VASP_default_INCAR": {"scf_nsoc": scf_defaults or {}, "scf_soc": scf_defaults or {}},
    }))
    mcp_server.main(config_path=str(config))
    return registered


def test_adsorption_analysis_is_one_visible_vasp_scf_call(tmp_path, monkeypatch):
    tools = _registered_tools(tmp_path, monkeypatch)
    assert "prepare_adsorption_analysis_scf" not in tools
    assert "submit_adsorption_analysis_scf" not in tools

    structure = tmp_path / "candidate.vasp"
    Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]).to(str(structure), fmt="poscar")
    preparation = tmp_path / "analysis_scf_manifest.json"
    calls = []

    def prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return {"manifest_path": str(preparation)}

    def load(path):
        calls.append(("load", path))
        return Path(path), {
            "structure_path": str(structure),
            "soc": False,
            "incar_tags": {"NSW": 0, "LAECHG": True},
            "kpoint_num": [4, 4, 1],
        }

    def submit(**kwargs):
        calls.append(("submit", kwargs))
        return {
            "success": True, "slurm_id": "42", "status": "pending",
            "error": None, "calculate_path": str(tmp_path / "calculation"),
        }

    def record(path, result):
        calls.append(("record", result))
        return {"success": True, "calculation_id": result["calculation_id"]}

    monkeypatch.setattr(mcp_server, "prepare_adsorption_analysis_scf", prepare)
    monkeypatch.setattr(mcp_server, "load_analysis_scf_submission", load)
    monkeypatch.setattr(mcp_server, "vasp_scf", submit)
    monkeypatch.setattr(mcp_server, "write_analysis_scf_submission", record)

    result = asyncio.run(tools["vasp_scf"](
        relaxation_manifest_path="/relaxations.json", analyses=["bader"]
    ))

    assert result["success"] is True
    assert [name for name, _ in calls] == ["prepare", "load", "submit", "record"]
    assert calls[0][1]["relaxation_manifest"] == "/relaxations.json"
    assert calls[2][1]["incar_dict"]["LAECHG"] is True


def test_adsorption_scf_rejects_conflicting_structure_input(tmp_path, monkeypatch):
    tools = _registered_tools(tmp_path, monkeypatch)
    result = asyncio.run(tools["vasp_scf"](
        relaxation_manifest_path="/relaxations.json",
        structure_path="/other.vasp",
    ))
    assert result["success"] is False
    assert "structure_path" in result["error"]


def test_scf_restart_for_bader_sets_charge_output_and_static_tags(tmp_path, monkeypatch):
    structure = Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]])
    relaxation_dir = tmp_path / "relaxation"
    _source_inputs(relaxation_dir)
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "completed",
        "structure": structure, "calculate_path": str(relaxation_dir),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []

    def submit(**kwargs):
        submitted.append(kwargs)
        return {
            "success": True, "slurm_id": "42", "status": "submitted",
            "error": None, "calculate_path": str(tmp_path / "scf"),
        }

    monkeypatch.setattr(mcp_server, "vasp_scf", submit)
    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", analyses=["bader"]
    ))

    assert result["success"] is True
    assert len(submitted) == 1
    assert submitted[0]["struct"] == structure
    assert submitted[0]["chgcar_path"] is None
    assert submitted[0]["source_potcar_path"] == str(relaxation_dir / "POTCAR")
    assert submitted[0]["incar_dict"]["LAECHG"] is True
    assert submitted[0]["incar_dict"]["LCHARG"] is True
    assert submitted[0]["incar_dict"]["NSW"] == 0
    assert submitted[0]["incar_dict"]["IBRION"] == -1


def test_scf_restart_for_bader_rejects_conflicting_overrides(tmp_path, monkeypatch):
    structure = Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]])
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "completed",
        "structure": structure, "calculate_path": str(tmp_path),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: submitted.append(kwargs))

    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", analyses=["bader"],
        incar_tags={"LAECHG": False},
    ))

    assert result["success"] is False
    assert "LAECHG" in result["error"]
    assert submitted == []


def test_bader_scf_does_not_restart_unfinished_relaxation(tmp_path, monkeypatch):
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "running",
        "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
        "calculate_path": str(tmp_path),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    def submit(**kwargs):
        submitted.append(kwargs)
        return {
            "success": True, "slurm_id": "42", "status": "submitted",
            "error": None, "calculate_path": str(tmp_path / "scf"),
        }
    monkeypatch.setattr(mcp_server, "vasp_scf", submit)

    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", analyses=["bader"]
    ))

    assert result["success"] is False
    assert "completed" in result["error"]
    assert submitted == []


def _source_inputs(directory):
    directory.mkdir(exist_ok=True)
    (directory / "POTCAR").write_text("   TITEL  = PAW_PBE Pd_pv 06Sep2000\n   ENMAX  = 271.098; ENMIN = 203.323 eV\n")
    (directory / "INCAR").write_text("ENCUT = 300\nISMEAR = 0\nSIGMA = 0.03\nISPIN = 2\n")
    (directory / "KPOINTS").write_text("Automatic kpoint scheme\n0\nGamma\n12 12 1\n")
    (directory / "CHGCAR").write_text("charge data")
    (directory / "WAVECAR").write_text("wave data")


def test_restart_inherits_source_inputs_without_reusing_charge_by_default(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "completed",
        "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
        "calculate_path": str(source),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []

    def submit(**kwargs):
        submitted.append(kwargs)
        return {"success": True, "slurm_id": "42", "status": "submitted",
                "error": None, "calculate_path": str(tmp_path / "scf")}

    monkeypatch.setattr(mcp_server, "vasp_scf", submit)
    result = asyncio.run(tools["vasp_scf"](restart_id="relax-1"))

    assert result["success"] is True
    assert submitted[0]["source_potcar_path"] == str(source / "POTCAR")
    assert submitted[0]["kpoints"].kpts == [(12, 12, 1)]
    assert submitted[0]["incar_dict"]["ENCUT"] == 300
    assert submitted[0]["incar_dict"]["ISPIN"] == 2
    assert submitted[0]["chgcar_path"] is None
    assert submitted[0]["wavecar_path"] is None
    assert submitted[0]["incar_dict"]["ICHARG"] == 2
    assert submitted[0]["incar_dict"]["ISTART"] == 0


def test_restart_rejects_charge_reuse_with_changed_potcar(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "completed",
        "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
        "calculate_path": str(source),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    def submit(**kwargs):
        submitted.append(kwargs)
        return {"success": True, "slurm_id": "42", "status": "submitted",
                "error": None, "calculate_path": str(tmp_path / "scf")}

    monkeypatch.setattr(mcp_server, "vasp_scf", submit)

    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", potcar_map={"Pd": "Pd"},
        incar_tags={"ICHARG": 1},
    ))

    assert result["success"] is False
    assert result["retryable"] is False
    assert "POTCAR" in result["error"]
    assert submitted == []


def test_restart_can_reuse_charge_with_matching_potcar(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "completed",
        "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
        "calculate_path": str(source),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []

    def submit(**kwargs):
        submitted.append(kwargs)
        return {"success": True, "slurm_id": "42", "status": "submitted",
                "error": None, "calculate_path": str(tmp_path / "scf")}

    monkeypatch.setattr(mcp_server, "vasp_scf", submit)
    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", incar_tags={"ICHARG": 1},
    ))

    assert result["success"] is True
    assert submitted[0]["chgcar_path"] == str(source / "CHGCAR")
    assert submitted[0]["wavecar_path"] is None


def test_charge_restart_rejects_spin_change(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "completed",
        "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
        "calculate_path": str(source),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: (submitted.append(kwargs) or {"success": True, "slurm_id": "42", "status": "submitted", "error": None, "calculate_path": str(tmp_path / "scf")}))

    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", incar_tags={"ICHARG": 1, "ISPIN": 1},
    ))

    assert result["success"] is False
    assert "ISPIN" in result["error"]
    assert submitted == []


def test_scf_rejects_ionic_motion_override(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "completed",
        "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
        "calculate_path": str(source),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: (submitted.append(kwargs) or {"success": True, "slurm_id": "42", "status": "submitted", "error": None, "calculate_path": str(tmp_path / "scf")}))

    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", incar_tags={"NSW": 20},
    ))

    assert result["success"] is False
    assert "NSW" in result["error"]
    assert submitted == []


def test_calculation_result_exposes_recorded_input_provenance(tmp_path, monkeypatch):
    provenance = {"potcar_titles": ["PAW_PBE Pd_pv 06Sep2000"],
                  "encut_eV": 300.0, "kpoint_mesh": [12, 12, 1]}
    records = {"scf-1": {"calc_type": "scf", "status": "completed",
                          "total_energy": -187.0, "input_provenance": provenance}}
    tools = _registered_tools(tmp_path, monkeypatch, records)

    result = asyncio.run(tools["read_calc_results_from_db"](["scf-1"]))

    assert result["scf-1"]["input_provenance"] == provenance


def test_changed_potcar_requires_explicit_electron_count_for_charged_source(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    with (source / "INCAR").open("a") as handle:
        handle.write("NELECT = 9\n")
    records = {"relax-1": {
        "calc_type": "relaxation", "status": "completed",
        "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
        "calculate_path": str(source),
    }}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: submitted.append(kwargs) or {
        "success": True, "slurm_id": "42", "status": "submitted",
        "error": None, "calculate_path": str(tmp_path / "scf")})

    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", potcar_map={"Pd": "Pd"},
    ))

    assert result["success"] is False
    assert "NELECT" in result["error"]
    assert submitted == []


def test_restart_preserves_explicit_source_precision_and_avoids_new_magnetism(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    (source / "INCAR").write_text("PREC = Normal\nADDGRID = True\nISPIN = 2\n")
    records = {"relax-1": {"calc_type": "relaxation", "status": "completed",
                           "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
                           "calculate_path": str(source)}}
    tools = _registered_tools(tmp_path, monkeypatch, records,
                              scf_defaults={"PREC": "Accurate", "MAGMOM": "5000*0", "EDIFF": 1e-7})
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: submitted.append(kwargs) or {
        "success": True, "slurm_id": "42", "status": "submitted", "error": None,
        "calculate_path": str(tmp_path / "scf")})

    result = asyncio.run(tools["vasp_scf"](restart_id="relax-1"))

    assert result["success"] is True
    assert submitted[0]["incar_dict"]["PREC"] == "Normal"
    assert submitted[0]["incar_dict"]["ADDGRID"] is True
    assert "MAGMOM" not in submitted[0]["incar_dict"]
    assert float(submitted[0]["incar_dict"]["EDIFF"]) == 1e-7


def test_restart_rejects_partial_potcar_map(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {"calc_type": "relaxation", "status": "completed",
                           "structure": Structure(Lattice.cubic(3), ["Pd", "H"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
                           "calculate_path": str(source)}}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: submitted.append(kwargs))

    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", potcar_map={"H": "H"}))

    assert result["success"] is False
    assert "Pd" in result["error"]
    assert submitted == []


def test_config_cannot_implicitly_enable_charge_restart(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {"calc_type": "relaxation", "status": "completed",
                           "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
                           "calculate_path": str(source)}}
    tools = _registered_tools(tmp_path, monkeypatch, records,
                              scf_defaults={"ICHARG": 1, "ISTART": 1})
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: submitted.append(kwargs) or {
        "success": True, "slurm_id": "42", "status": "submitted", "error": None,
        "calculate_path": str(tmp_path / "scf")})

    result = asyncio.run(tools["vasp_scf"](restart_id="relax-1"))

    assert result["success"] is True
    assert submitted[0]["incar_dict"]["ICHARG"] == 2
    assert submitted[0]["incar_dict"]["ISTART"] == 0
    assert submitted[0]["chgcar_path"] is None
    assert submitted[0]["wavecar_path"] is None


def test_lowercase_incar_override_is_normalized_and_audited(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {"calc_type": "relaxation", "status": "completed",
                           "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
                           "calculate_path": str(source)}}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: submitted.append(kwargs) or {
        "success": True, "slurm_id": "42", "status": "submitted", "error": None,
        "calculate_path": str(tmp_path / "scf")})

    result = asyncio.run(tools["vasp_scf"](
        restart_id="relax-1", incar_tags={"encut": 400, "icharg": 1}))

    assert result["success"] is True
    assert submitted[0]["incar_dict"]["ENCUT"] == 400
    assert submitted[0]["incar_dict"]["ICHARG"] == 1
    assert result["input_provenance"]["explicit_changes"]["incar_tags"] == {"ENCUT": 400, "ICHARG": 1}


@pytest.mark.parametrize("override", [{"ICHARG": "invalid"}, {"ISTART": "invalid"}])
def test_restart_rejects_malformed_restart_flags(tmp_path, monkeypatch, override):
    source = tmp_path / "rlx"
    _source_inputs(source)
    records = {"relax-1": {"calc_type": "relaxation", "status": "completed",
                           "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
                           "calculate_path": str(source)}}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: submitted.append(kwargs))

    result = asyncio.run(tools["vasp_scf"](restart_id="relax-1", incar_tags=override))

    assert result["success"] is False
    assert result["error_code"] == "INVALID_SCF_INPUT"
    assert submitted == []


def test_scf_recovers_completed_source_with_expired_scheduler_id(tmp_path, monkeypatch):
    source = tmp_path / "rlx"
    _source_inputs(source)
    record = {"calc_type": "relaxation", "status": "unknown", "success": False,
              "error": "Invalid job id specified", "slurm_id": "95",
              "structure": Structure(Lattice.cubic(3), ["Pd"], [[0, 0, 0]]),
              "calculate_path": str(source)}
    records = {"relax-1": record}
    tools = _registered_tools(tmp_path, monkeypatch, records)
    monkeypatch.setattr(mcp_server, "recover_completed_vasp_result", lambda data: {
        "status": "completed", "success": True, "error": None,
        "total_energy": -187.25900531})
    submitted = []
    monkeypatch.setattr(mcp_server, "vasp_scf", lambda **kwargs: submitted.append(kwargs) or {
        "success": True, "slurm_id": "96", "status": "submitted", "error": None,
        "calculate_path": str(tmp_path / "scf")})

    result = asyncio.run(tools["vasp_scf"](restart_id="relax-1"))

    assert result["success"] is True
    assert records["relax-1"]["status"] == "completed"
    assert records["relax-1"]["total_energy"] == -187.25900531
    assert submitted[0]["source_potcar_path"] == str(source / "POTCAR")
