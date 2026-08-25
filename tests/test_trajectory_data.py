from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import Trajectory

from vaspilot.server.trajectory_data import load_trajectory_data


def test_loads_frames_energies_forces_and_roles(tmp_path):
    path = tmp_path / "relaxation.traj"
    with Trajectory(path, "w") as trajectory:
        for step in range(2):
            atoms = Atoms(
                "PtCO",
                positions=[[0, 0, 0], [0, 0, 2 - step * 0.1], [0, 0, 3]],
                cell=[5, 5, 15],
                pbc=True,
                tags=[0, 2, 2],
            )
            atoms.set_constraint(FixAtoms(indices=[0]))
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=-step,
                forces=[[0, 0, 0], [0, 0, 0.2 - step * 0.1], [0, 0, 0]],
            )
            trajectory.write(atoms)

    result = load_trajectory_data(str(path))

    assert result["frame_count"] == 2
    assert result["symbols"] == ["Pt", "C", "O"]
    assert result["fixed_atom_indices"] == [0]
    assert result["adsorbate_atom_indices"] == [1, 2]
    assert result["adsorbate_indices_inferred"] is False
    assert result["frames"][1]["energy_eV"] == -1.0
    assert result["frames"][0]["max_force_eV_per_A"] == 0.2
