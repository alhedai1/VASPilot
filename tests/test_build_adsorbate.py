from pathlib import Path

from ase.io import read

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

    output_path = Path(result["adsorbate_structure_path"])
    assert output_path.is_file()
    assert read(output_path).get_chemical_symbols() == ["O", "C"]


def test_normalizes_adsorbate_name(tmp_path):
    result = build_adsorbate(" h2o ", str(tmp_path))

    assert result["success"] is True
    assert result["adsorbate"] == "H2O"
    assert result["binding_atom_elements"] == ["O"]


def test_rejects_unknown_adsorbate(tmp_path):
    result = build_adsorbate("CH4", str(tmp_path))

    assert result["success"] is False
    assert "Supported adsorbates" in result["error"]
