
1. structure analysis
Search for the crystal structure of MoS2 and analyze its lattice parameters,
space group, composition, and number of atoms.
Do not run any VASP calculations.
Validate that all reported structure files exist.

tests:
search_materials_project
analyze_crystal_structure
check_files_exist

modified prompt wording -> search/analyze works, llm fails to respond to check_files_exist 3 times
llm failures probably due to ollama/crewai integration
try:
    Running Qwen3 in non-thinking mode, especially for the validation agent. Add /no_think to that agent’s instructions or use a non-thinking model.
    Explicitly instructing the validation agent: “After the file-check tool returns, immediately provide a non-empty final answer. Do not call the same tool again.”
    Keeping the validation task’s expected output very simple and explicit.
    If empty responses continue, using a different Ollama instruct model for the manager and validation roles. Tool execution can remain unchanged.

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


COMMANDS
vaspilot_mcp --config examples/1.Basic/configs/mcp_config.yaml --port 8933
vaspilot_quart \
  --config examples/1.Basic/configs/crew_config_en.yaml \
  --port 51293 \
  --work-dir examples/1.Basic/crew_server/work-local-after-expected-output-fix-3 \
  --allow-path examples/1.Basic

notes
disabled memory to test agents and tools with simple missions
enable memory later and fix version issue: chroma crewai issue


why does crystal agent fetch many variants (for example 4 for 2H phase of MoS2)?