import json
import sys

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Chgcar, Poscar

from vaspilot.tools.mcp.bader_analysis import parse_acf, run_bader_analysis


def test_bader_runner_sums_core_and_valence_and_parses_atoms(tmp_path):
    scf_dir = tmp_path / "scf"
    scf_dir.mkdir()
    structure = Structure(Lattice.cubic(4), ["H"], [[0, 0, 0]])
    for name, value in (("CHGCAR", 3), ("AECCAR0", 1), ("AECCAR2", 2)):
        Chgcar(Poscar(structure), {"total": np.full((2, 2, 2), value)}).write_file(scf_dir / name)

    executable = tmp_path / "fake_bader"
    executable.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "assert sys.argv[1].endswith('CHGCAR')\n"
        "assert sys.argv[2:] == ['-ref', 'CHGCAR_sum']\n"
        "Path('ACF.dat').write_text("
        "'# X Y Z CHARGE MIN DIST ATOMIC VOL\\n'"
        "'1 0 0 0 0.75 0.1 10\\n'"
        "'--------------------------------------------------\\n'"
        "'VACUUM CHARGE: 0.0\\nNUMBER OF ELECTRONS: 0.75\\n')\n"
    )
    executable.chmod(0o755)

    result = run_bader_analysis(scf_dir, tmp_path / "analysis", str(executable))

    assert result["success"] is True
    assert result["atoms"] == [{"index": 1, "electrons": 0.75}]
    assert result["acf_path"] == str(tmp_path / "analysis" / "ACF.dat")
    summed = Chgcar.from_file(tmp_path / "analysis" / "CHGCAR_sum")
    assert np.allclose(summed.data["total"], 3)


def test_parse_acf_rejects_missing_atom(tmp_path):
    acf = tmp_path / "ACF.dat"
    acf.write_text("# X Y Z CHARGE MIN DIST ATOMIC VOL\n")
    with pytest.raises(ValueError, match="atom count"):
        parse_acf(acf, expected_atoms=1)
