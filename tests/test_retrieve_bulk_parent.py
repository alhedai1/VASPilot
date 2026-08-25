from pathlib import Path
from types import SimpleNamespace

from vaspilot.tools.mcp import struct_tools


class FakeStructure:
    def __init__(self, formula):
        self.composition = SimpleNamespace(reduced_formula=formula)

    def to(self, filename, fmt):
        assert fmt == "poscar"
        Path(filename).write_text("fake POSCAR\n")


def material(material_id, hull, formation, formula="Pt"):
    return SimpleNamespace(
        material_id=material_id,
        energy_above_hull=hull,
        formation_energy_per_atom=formation,
        structure=FakeStructure(formula),
    )


def install_fake_client(monkeypatch, materials, calls):
    class FakeSummary:
        def search(self, **kwargs):
            calls.append(kwargs)
            return materials

    class FakeMPRester:
        def __init__(self, api_key):
            self.materials = SimpleNamespace(summary=FakeSummary())

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(struct_tools, "MPRester", FakeMPRester)


def test_formula_selects_lowest_hull_then_formation_energy(monkeypatch, tmp_path):
    calls = []
    install_fake_client(
        monkeypatch,
        [
            material("mp-3", 0.1, -1.0),
            material("mp-2", 0.0, -0.5),
            material("mp-1", 0.0, -0.8),
        ],
        calls,
    )

    result = struct_tools.retrieve_bulk_parent(
        api_key="test", download_path=str(tmp_path), formula="Pt"
    )

    assert calls == [{"formula": "Pt"}]
    assert result["success"] is True
    assert result["material_id"] == "mp-1"
    assert result["structure_kind"] == "bulk_parent"
    assert result["selection_method"] == "lowest_energy_above_hull_then_formation_energy"
    assert Path(result["structure_path"]).is_file()


def test_material_id_takes_precedence_over_formula(monkeypatch, tmp_path):
    calls = []
    install_fake_client(
        monkeypatch,
        [material("mp-126", 0.0, -1.0)],
        calls,
    )

    result = struct_tools.retrieve_bulk_parent(
        api_key="test",
        download_path=str(tmp_path),
        formula="ignored",
        material_id="mp-126",
    )

    assert calls == [{"material_ids": ["mp-126"]}]
    assert result["success"] is True
    assert result["selection_method"] == "explicit_material_id"


def test_requires_formula_or_material_id(tmp_path):
    result = struct_tools.retrieve_bulk_parent(
        api_key="test", download_path=str(tmp_path)
    )

    assert result == {
        "success": False,
        "error": "Either material_id or formula must be provided",
    }
