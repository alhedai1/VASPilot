from pathlib import Path

from ase import Atoms
from ase.io import read, write

from vaspilot.tools.mcp.struct_tools import build_adsorbate


def test_builds_co_with_carbon_binding_atom(tmp_path):
    result = build_adsorbate("CO", str(tmp_path))

    assert result["success"] is True
    assert result["structure_kind"] == "adsorbate"
    assert result["adsorbate"] == "CO"
    assert result["chemical_symbols"] == ["O", "C"]
    assert result["binding_atom_indices"] == [1]
    assert result["binding_atom_elements"] == ["C"]
    assert result["binding_indexing"] == "zero_based"
    assert result["charge"] == 0
    assert result["multiplicity"] == 1
    assert len(result["artifact_id"]) == 16
    assert result["artifact_id"] in Path(result["adsorbate_structure_path"]).name

    output_path = Path(result["adsorbate_structure_path"])
    assert output_path.is_file()
    assert read(output_path).get_chemical_symbols() == ["O", "C"]


def test_normalizes_adsorbate_name(tmp_path):
    result = build_adsorbate(" H2O ", str(tmp_path))

    assert result["success"] is True
    assert result["adsorbate"] == "H2O"
    assert result["binding_atom_elements"] == ["O"]

    co2 = build_adsorbate("CO2", str(tmp_path))
    assert co2["binding_atom_indices"] == [0]
    assert co2["binding_atom_elements"] == ["C"]


def test_distinguishes_element_case_from_molecule(tmp_path):
    cobalt = build_adsorbate("Co", str(tmp_path), charge=0, multiplicity=2)
    carbon_monoxide = build_adsorbate("CO", str(tmp_path))
    silver = build_adsorbate("Ag", str(tmp_path))

    assert cobalt["success"] is True
    assert cobalt["chemical_symbols"] == ["Co"]
    assert cobalt["binding_atom_indices"] == [0]
    assert cobalt["construction_source"] == "ASE elemental atom"
    assert carbon_monoxide["chemical_symbols"] == ["O", "C"]
    assert carbon_monoxide["construction_source"] == "ASE molecule library"
    assert silver["success"] is True
    assert silver["chemical_symbols"] == ["Ag"]
    assert silver["charge"] is None
    assert silver["multiplicity"] is None
    assert silver["electronic_state_specified"] is False


def test_uncurated_ase_adsorbate_allows_unspecified_electronic_state(tmp_path):
    result = build_adsorbate("CH4", str(tmp_path))

    assert result["success"] is False
    assert "binding_atom_indices" in result["error"]

    result = build_adsorbate("CH4", str(tmp_path), binding_atom_indices=[0])
    assert result["success"] is True
    assert result["binding_atom_elements"] == ["C"]
    assert result["charge"] is None
    assert result["multiplicity"] is None
    assert result["construction_metadata"]["charge_source"] == "unspecified"


def test_builds_file_backed_adsorbate_with_provenance(tmp_path):
    source = tmp_path / "custom.xyz"
    write(source, Atoms("CN", positions=[[0, 0, 0], [0, 0, 1.2]]))

    result = build_adsorbate(
        "CN",
        str(tmp_path),
        source_structure_path=str(source),
        binding_atom_indices=[1],
        charge=-1,
        multiplicity=1,
    )

    assert result["success"] is True
    assert result["construction_source"] == "user_structure_file"
    assert result["construction_metadata"]["source_sha256"]
    assert result["binding_atom_elements"] == ["N"]
    assert result["charge"] == -1


def test_parameters_change_artifact_identity(tmp_path):
    carbon = build_adsorbate("CO", str(tmp_path), binding_atom_indices=[1])
    oxygen = build_adsorbate("CO", str(tmp_path), binding_atom_indices=[0])

    assert carbon["artifact_id"] != oxygen["artifact_id"]
    assert carbon["adsorbate_structure_path"] != oxygen["adsorbate_structure_path"]


def test_rejects_invalid_binding_indices(tmp_path):
    result = build_adsorbate("CO", str(tmp_path), binding_atom_indices=[2])
    assert result["success"] is False
    assert "valid zero-based" in result["error"]


def test_reports_missing_ase_geometry_as_non_retryable(tmp_path):
    result = build_adsorbate("MoS2", str(tmp_path))

    assert result == {
        "success": False,
        "error_code": "adsorbate_geometry_unavailable",
        "error": (
            "ASE has no molecular geometry named 'MoS2'. "
            "Provide source_structure_path and binding_atom_indices."
        ),
        "retryable": False,
        "required_inputs": ["source_structure_path", "binding_atom_indices"],
    }
