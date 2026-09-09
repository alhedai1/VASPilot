import json
from pathlib import Path

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
import pytest

from vaspilot.tools.mcp.adsorption_energy import (
    calculate_adsorption_energies,
    prepare_adsorption_energy_calculations,
)


def _structures(tmp_path: Path):
    slab = Atoms("Pt2", positions=[[0, 0, 5], [2, 0, 5]], cell=[8, 8, 20], pbc=True)
    slab.set_constraint(FixAtoms(indices=[0]))
    adsorbate = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.1]])
    slab_path, adsorbate_path = tmp_path / "slab.vasp", tmp_path / "CO.xyz"
    write(slab_path, slab, format="vasp")
    write(adsorbate_path, adsorbate)
    candidates = []
    for index in range(2):
        atoms = slab + adsorbate
        atoms.positions[-2:] += [1 + index, 1, 7]
        atoms.set_constraint(FixAtoms(indices=[0]))
        path = tmp_path / f"candidate_{index:03d}_relaxed.vasp"
        write(path, atoms, format="vasp")
        candidates.append(path)
    return slab_path, adsorbate_path, candidates


def test_prepares_top_k_consistent_calculation_set(tmp_path):
    slab, adsorbate, candidates = _structures(tmp_path)
    manifest = {
        "artifact_id": "relax-1",
        "results": [
            {"candidate_id": "candidate_000", "rank": 2, "status": "converged", "final_structure_path": str(candidates[0]), "geometry_diagnostics": {}},
            {"candidate_id": "candidate_001", "rank": 1, "status": "converged", "final_structure_path": str(candidates[1]), "geometry_diagnostics": {}},
        ],
    }
    manifest_path = tmp_path / "relaxation_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = prepare_adsorption_energy_calculations(
        manifest_path, str(slab), str(adsorbate), str(tmp_path), top_k=1,
        method_settings={"xc": "PBE", "encut_eV": 450},
    )

    assert result["success"] is True
    assert result["selected_candidate_count"] == 1
    assert [item["role"] for item in result["calculations"]] == [
        "clean_slab", "isolated_adsorbate", "adsorbed_candidate"
    ]
    assert result["calculations"][-1]["candidate_id"] == "candidate_001"
    assert result["method_settings"]["dipole_correction"] is True
    assert result["method_settings"]["kpoint_policy"] == "automatic_density_40"
    assert Path(result["manifest_path"]).is_file()
    molecule = read(result["calculations"][1]["structure_path"])
    assert molecule.cell.lengths() == pytest.approx([20, 20, 20])


def test_rejects_candidate_with_wrong_composition(tmp_path):
    slab, adsorbate, candidates = _structures(tmp_path)
    write(candidates[0], read(slab), format="vasp")
    manifest = {"results": [{"candidate_id": "bad", "rank": 1, "status": "converged", "final_structure_path": str(candidates[0]), "geometry_diagnostics": {}}]}
    with pytest.raises(ValueError, match="not clean slab plus adsorbate"):
        prepare_adsorption_energy_calculations(manifest, str(slab), str(adsorbate), str(tmp_path))


def test_requires_electronic_settings_for_charged_or_open_shell_adsorbate(tmp_path):
    slab, adsorbate, candidates = _structures(tmp_path)
    manifest = {"results": [{"candidate_id": "a", "rank": 1, "status": "converged", "final_structure_path": str(candidates[0]), "geometry_diagnostics": {}}]}
    with pytest.raises(ValueError, match="nelect"):
        prepare_adsorption_energy_calculations(
            manifest, str(slab), str(adsorbate), str(tmp_path), adsorbate_charge=1,
        )
    with pytest.raises(ValueError, match="spin_polarized"):
        prepare_adsorption_energy_calculations(
            manifest, str(slab), str(adsorbate), str(tmp_path), adsorbate_multiplicity=2,
        )


def test_calculates_ranks_and_separates_corrections(tmp_path):
    reference = {"converged": True, "method_signature": "method-1"}
    result = calculate_adsorption_energies(
        [
            {**reference, "candidate_id": "b", "calculation_id": "b", "energy_eV": -14.0},
            {**reference, "candidate_id": "a", "calculation_id": "a", "energy_eV": -15.0},
        ],
        {**reference, "calculation_id": "slab", "energy_eV": -10.0},
        {**reference, "calculation_id": "mol", "energy_eV": -3.0},
        str(tmp_path),
        corrections={"a": {"zpe": 0.2, "thermal": 0.1}},
    )
    assert result["best_candidate_id"] == "a"
    best = result["results"][0]
    assert best["electronic_adsorption_energy_eV"] == pytest.approx(-2.0)
    assert best["total_correction_eV"] == pytest.approx(0.3)
    assert best["corrected_adsorption_energy_eV"] == pytest.approx(-1.7)
    assert Path(result["manifest_path"]).is_file()


@pytest.mark.parametrize("change", [
    {"converged": False}, {"method_signature": "different"},
])
def test_rejects_incompatible_or_unconverged_energy_records(tmp_path, change):
    reference = {"converged": True, "method_signature": "method-1"}
    candidate = {**reference, "candidate_id": "a", "energy_eV": -15.0, **change}
    with pytest.raises(ValueError):
        calculate_adsorption_energies(
            [candidate], {**reference, "energy_eV": -10.0},
            {**reference, "energy_eV": -3.0}, str(tmp_path),
        )
