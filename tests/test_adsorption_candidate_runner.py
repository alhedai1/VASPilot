from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write
from ase.io import read
import numpy as np
import pytest

from vaspilot.scripts.adsorption_candidate_runner import (
    CandidateBudgetExceeded,
    _geometry_diagnostics,
    _wrap_candidate_in_plane,
    generate_candidates,
    validate_generation_size,
)


def test_geometry_diagnostics_is_independent_of_output_directory():
    atoms = Atoms(
        "PtCO",
        positions=[[0, 0, 0], [0, 0, 2], [0, 0, 3.1]],
        cell=[10, 10, 20],
        pbc=True,
    )
    result = _geometry_diagnostics(atoms, 1, np.array([0.0, 0.0, 1.0]))
    assert result["minimum_adsorbate_slab_distance_A"] == pytest.approx(2.0)


def test_caps_heuristic_sites_and_records_provenance(tmp_path, monkeypatch):
    slab_path = tmp_path / "slab.vasp"
    adsorbate_path = tmp_path / "CO.xyz"
    slab = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]], cell=[4, 4, 12])
    slab.set_constraint(FixAtoms(indices=[0]))
    write(slab_path, slab, format="vasp")
    write(adsorbate_path, Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.1]]))

    class FakeConfig:
        def __init__(self, slab, adsorbate, **kwargs):
            self.sites = list(range(12))
            combined = slab.slab_atoms + adsorbate.adsorbate_atoms
            self.atoms_list = [combined.copy() for _ in range(24)]
            self.metadata_list = [{"site": index // 2} for index in range(24)]

    import fairchem.data.oc.core as core

    monkeypatch.setattr(core, "Slab", lambda slab_atoms, **kwargs: SimpleNamespace(slab_atoms=slab_atoms))
    monkeypatch.setattr(
        core,
        "Adsorbate",
        lambda adsorbate_atoms, **kwargs: SimpleNamespace(adsorbate_atoms=adsorbate_atoms),
    )
    monkeypatch.setattr(core, "AdsorbateSlabConfig", FakeConfig)

    result = generate_candidates(
        slab_path, adsorbate_path, [0], 10, 2, random_seed=7
    )

    assert result["candidate_count"] == 20
    assert result["generation_metadata"]["available_site_count"] == 12
    assert result["generation_metadata"]["selected_site_count"] == 10
    assert result["generation_metadata"]["random_seed"] == 7
    assert result["generation_metadata"]["max_generated_candidates"] == 1000
    assert result["generation_metadata"]["requested_candidate_upper_bound"] == 20
    assert result["generation_metadata"]["generator_schema_version"] == 3
    assert result["artifact_id"] in result["candidate_directory"]
    assert result["manifest_path"].endswith("candidate_manifest.json")
    assert Path(result["manifest_path"]).is_file()
    candidate = result["candidates"][0]
    scaled = read(candidate["structure_path"]).get_scaled_positions(wrap=False)
    assert (scaled[:, :2] >= -1e-12).all()
    assert (scaled[:, :2] < 1 + 1e-12).all()
    assert candidate["geometry_diagnostics"]["slab_in_plane_coordinates_wrapped"] is True
    assert candidate["geometry_diagnostics"][
        "adsorbate_periodic_image_selected_as_rigid_unit"
    ] is True


def test_wraps_boundary_crossing_adsorbate_as_rigid_unit():
    atoms = Atoms(
        "PtCO",
        scaled_positions=[[-0.2, 1.2, 0.4], [0.95, 0.5, 0.7], [1.05, 0.5, 0.7]],
        cell=[10, 10, 20],
        pbc=True,
    )
    bond_before = atoms.positions[2] - atoms.positions[1]

    _wrap_candidate_in_plane(atoms, slab_atom_count=1, binding_atom_indices=[1])

    scaled = atoms.get_scaled_positions(wrap=False)
    assert scaled[0, :2] == pytest.approx([0.8, 0.2])
    assert scaled[2, :2] == pytest.approx([0.05, 0.5])
    assert atoms.positions[2] - atoms.positions[1] == pytest.approx(bond_before)


def test_rejects_unknown_placement_mode(tmp_path):
    with pytest.raises(ValueError, match="Unsupported placement_mode"):
        generate_candidates(
            tmp_path / "missing.vasp",
            tmp_path / "missing.xyz",
            [0],
            10,
            1,
            placement_mode="invalid",
        )


def test_accepts_generation_size_at_and_above_100_sites():
    validate_generation_size(500, 2, 1000)
    validate_generation_size(100, 10, 1000)


def test_rejects_exceeded_candidate_budget():
    with pytest.raises(CandidateBudgetExceeded, match="exceeds"):
        validate_generation_size(500, 10, 1000)


@pytest.mark.parametrize(
    "values",
    [(0, 1, 1000), (1, 0, 1000), (1, 1, 0), (True, 1, 1000)],
)
def test_rejects_invalid_generation_sizes(values):
    with pytest.raises(ValueError, match="positive integer"):
        validate_generation_size(*values)
