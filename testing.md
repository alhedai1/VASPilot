
1. structure analysis
Search for the crystal structure of MoS2 and analyze its lattice parameters,
space group, composition, and number of atoms.
Do not run any VASP calculations.
Validate that all reported structure files exist.

tests:
search_materials_project
analyze_crystal_structure
check_files_exist

2. Structure manipulation
Search for the crystal structure of MoS2, create a 2×2×1 supercell,
and analyze the resulting structure.
Do not run any VASP calculations.
Validate that the generated structure file exists.

tests:
make_supercell

3. Symmetrization
Search for a slightly distorted crystal structure, symmetrize it,
and report the space group before and after symmetrization.
Do not run VASP calculations.

tests:
symmetrize_structure

4. Structure creation
Create a conventional rocksalt NaCl crystal structure with a lattice
parameter of 5.64 Å. Analyze it and verify that the output file exists.
Do not run VASP calculations.

tests:
create_crystal_structure

5. controlled vasp test
Search for the primitive silicon crystal structure and perform only a
VASP relaxation using the default settings. Do not run SCF, NSCF,
band-structure, DOS, or plotting tasks. Report the calculation ID and wait
for the relaxation to finish. Validate the calculation result.



notes
disabled memory to test agents and tools with simple missions
enable memory later and fix version issue: chroma crewai issue