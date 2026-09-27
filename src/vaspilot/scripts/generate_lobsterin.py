from pymatgen.io.lobster import Lobsterin
from pathlib import Path

# cwd = Path.cwd()
# print(cwd)

lobsterin = Lobsterin.standard_calculations_from_vasp_files(
    POSCAR_input="POSCAR",
    INCAR_input="INCAR",
    POTCAR_input="POTCAR",
    Vasprun_output="vasprun.xml",
    option="standard",
)

lobsterin.write_lobsterin("lobsterin")