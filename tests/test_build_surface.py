from pathlib import Path

import numpy as np
import pytest

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from vaspilot.tools.mcp.struct_tools import build_surface


@pytest.fixture
def bulk_path(tmp_path):
    bulk = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(3.92), ["Pt"], [[0, 0, 0]]
    )
    bulk_path = tmp_path / "Pt.vasp"
    bulk.to(filename=bulk_path, fmt="poscar")
    return bulk_path


def test_builds_default_constrained_pt_111_slab(bulk_path):
    result = build_surface(str(bulk_path), [1, 1, 1])

    assert result["success"] is True
    assert result["structure_kind"] == "surface_slab"
    assert result["miller_index"] == [1, 1, 1]
    assert result["num_atomic_layers"] >= 3
    assert result["estimated_vacuum_thickness_A"] >= 14.0
    assert result["fixed_atom_indices"]
    assert result["surface_metadata"]["lateral_supercell"] == [4, 4]
    assert result["surface_metadata"]["construction_parameters"]["orthogonalize_c"] is False
    assert result["surface_metadata"]["builder_schema_version"] == 2
    assert len(result["artifact_id"]) == 16
    assert result["artifact_id"] in Path(result["slab_structure_path"]).name

    slab_path = Path(result["slab_structure_path"])
    assert slab_path.is_file()
    slab = Poscar.from_file(slab_path).structure
    assert np.all(slab.frac_coords[:, :2] >= -1e-12)
    assert np.all(slab.frac_coords[:, :2] < 1 + 1e-12)
    selective_dynamics = slab.site_properties["selective_dynamics"]
    assert len(slab) == result["num_atoms"]
    assert any(list(flags) == [False, False, False] for flags in selective_dynamics)
    assert any(list(flags) == [True, True, True] for flags in selective_dynamics)
    assert [i for i, flags in enumerate(selective_dynamics) if not any(flags)] == result["fixed_atom_indices"]
    assert {result["atomic_layer_indices"][i] for i in result["fixed_atom_indices"]} == {0, 1}


def test_custom_general_surface_has_orthogonal_c_and_unique_artifact(bulk_path):
    kwargs = dict(
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        lateral_supercell=[2, 3],
        fixed_bottom_layers=1,
        orthogonalize_c=True,
    )
    result = build_surface(str(bulk_path), [2, 1, 0], **kwargs)
    other = build_surface(str(bulk_path), [2, 1, 0], **{**kwargs, "min_vacuum_size": 13.0})

    assert result["success"] is True
    assert result["artifact_id"] != other["artifact_id"]
    assert result["slab_structure_path"] != other["slab_structure_path"]
    assert result["estimated_vacuum_thickness_A"] >= 11.0
    slab = Poscar.from_file(result["slab_structure_path"]).structure
    normal = np.asarray(result["surface_normal_cartesian"])
    assert np.linalg.norm(np.cross(slab.lattice.matrix[2], normal)) < 1e-8
    assert {result["atomic_layer_indices"][i] for i in result["fixed_atom_indices"]} == {0}


def test_selects_explicit_termination(bulk_path):
    default = build_surface(str(bulk_path), [1, 1, 1])
    indexed = build_surface(
        str(bulk_path), [1, 1, 1], termination_policy="index", termination_index=0
    )

    assert indexed["success"] is True
    assert indexed["surface_metadata"]["selected_termination_index"] == 0
    assert indexed["surface_metadata"]["termination_selection"] == "index"
    assert indexed["artifact_id"] != default["artifact_id"]


def test_rejects_zero_miller_index(tmp_path):
    bulk_path = tmp_path / "unused.vasp"
    bulk_path.write_text("not read")

    result = build_surface(str(bulk_path), [0, 0, 0])

    assert result["success"] is False
    assert "cannot be [0, 0, 0]" in result["error"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_slab_size": 0}, "positive numbers"),
        ({"lateral_supercell": [2, 0]}, "two positive integers"),
        ({"fixed_bottom_layers": -1}, "non-negative integer"),
        ({"termination_policy": "index"}, "termination_index"),
        ({"termination_index": 0}, "only be used"),
    ],
)
def test_rejects_invalid_parameters(bulk_path, kwargs, message):
    result = build_surface(str(bulk_path), [1, 1, 1], **kwargs)
    assert result["success"] is False
    assert message in result["error"]
