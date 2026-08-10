# MoS2 structure-analysis test
Date: 2026-08-04

## Goal
Test whether VASPilot can retrieve and analyze a crystal structure
without running VASP.

## Prompt 1
Search for the crystal structure of MoS2 and analyze its lattice
parameters, space group, composition, and number of atoms.
Do not run any VASP calculations.
Validate that all reported structure files exist.

## Observed behavior
Tools:
- search_materials_project
- did NOT use analyze_crystal_structure

Result:
- Structure was downloaded.
- Agent said that lattice parameters, space group, and atom count
  could be obtained from the file instead of actually obtaining them.

## Prompt 2
Search for the crystal structure of MoS2. Analyze its lattice
parameters, space group, composition, and number of atoms, using
the necessary tools, and include the results in the output.
Do not run any VASP calculations.
Validate that all reported structure files exist.

## Observed behavior
Tools:
- search_materials_project
- analyze_crystal_structure
- check_files_exist

Result:
- Requested structure properties were explicitly returned.
- File existence was validated.
- No VASP calculation was run.

## Observation
Adding "using the necessary tools, and include the results in the output"
changed the agent's behavior.

## Possible issue
Tool-use reliability depends too strongly on prompt wording.
The agent should infer that "analyze lattice parameters..." requires
analyze_crystal_structure without the user explicitly asking it
to use tools.

## Possible improvement
Improve agent/task/tool descriptions or orchestration so that requested
properties automatically trigger the appropriate analysis tool.

## Follow-up experiments
- Repeat Prompt 1 multiple times.
- Repeat Prompt 2 multiple times.
- Try other materials.
- Check whether behavior changes across models.


## Validation-agent retry behavior

Observation:
result_validation_agent called check_files_exist three times.

All three tool calls successfully returned:
{"<path>": true}

However, after the first two tool calls, the subsequent LLM call failed with:

"Received None or empty response from LLM call."
"Invalid response from LLM call - None or empty."

The agent was restarted/retried, causing the same tool to execute again.

The third attempt successfully generated the final validation response.

Possible issue:
LLM/tool-call integration failure causes unnecessary repeated tool execution.

Potential improvement:
- Retry only the failed LLM generation rather than rerunning an already
  successful/idempotent tool.
- Cache successful tool results across retries.
- Investigate why the configured LLM sometimes returns None/empty responses.


## Materials Project versus VASPilot visualization

### Observation

A structure downloaded by VASPilot can look different from the structure shown
on the Materials Project website even when both use the same Materials Project
ID. The VASPilot view may contain fewer visible spheres and may make the atoms
look like separate small groups rather than a continuous periodic crystal.

The clearest example was the 2H phase of MoS2, `mp-2815`. In VASPilot/JSmol it
looked like two groups, each containing one Mo sphere and two S spheres. The
Materials Project viewer showed many more spheres for the same ID.

### What was verified

The downloaded file was:

`examples/1.Basic/mcp/downloads/mp-2815_MoS2.vasp`

Its contents were checked with pymatgen and found to be consistent with
2H-MoS2:

- Full cell composition: `Mo2S4`
- Reduced formula: `MoS2`
- Number of sites explicitly stored in the unit cell: 6
- Species: 2 Mo and 4 S
- Space group: `P6_3/mmc` (No. 194)
- Lattice parameters: `a = b = 3.192238 A`, `c = 13.378294 A`

Six atoms are expected for this 2H-MoS2 unit cell. The two apparent groups are
the two MoS2 formula units/layers explicitly present in the file.
 
`mp-1434` also demonstrated that primitive and conventional representations
can have different atom counts while describing the same periodic material.
Its downloaded primitive cell contained 3 atoms, whereas its conventional
hexagonal representation contained 9 atoms.

### Cause

The downloaded structure is not necessarily wrong. There are two separate
representation effects:

1. VASPilot writes the structure returned by the Materials Project API directly
   to POSCAR format. It does not convert it to the same conventional-cell
   representation used by the Materials Project website.
2. VASPilot's JSmol view displays primarily the atoms explicitly stored in that
   unit cell. The Materials Project viewer displays surrounding periodic images,
   so it can show many more spheres and more complete coordination environments.

JSmol also infers bonds using its own distance rules. For example, Mo in
2H-MoS2 has sixfold S coordination, but several coordinating S atoms can lie in
neighboring periodic images. If those images are not displayed, JSmol can make
Mo appear connected to only the two S atoms visible in the same grouping.

### Concise conclusion

The main issue is limited/different periodic visualization, not an incorrect
Materials Project structure. VASPilot/JSmol shows one explicit unit cell, while
Materials Project commonly shows a conventional cell plus periodic neighboring
copies. Consequently, the pictures and visible sphere counts differ even though
the underlying periodic crystal can be equivalent.

Different atom counts are normal only when comparing primitive, conventional,
or replicated cells. The reduced formula, elemental ratio, symmetry-equivalent
structure, and volume per formula unit should remain consistent.

### Important distinction from LLM output

The atom count reported by the agent must be checked against the actual
downloaded file or the output of `analyze_crystal_structure`. In an earlier run,
the agent reported 54 atoms for `mp-1434`, although its downloaded POSCAR
contained 3 atoms and that exact file had not been analyzed. That was an
unsupported LLM-generated result, not a JSmol or Materials Project atom count.

### Relevant implementation locations

- Download/write behavior: `src/vaspilot/tools/mcp/struct_tools.py`, in
  `search_materials_project`; it uses `material_data.structure` and writes it
  directly using `structure.to(..., fmt="poscar")`.
- Visualization behavior:
  `src/vaspilot/server/quart_server/templates/task_detail.html`, in
  `loadVaspStructure`; JSmol loads the POSCAR and enables the unit-cell display
  without explicitly requesting a periodic supercell/packed view.

### Possible follow-up

- Add an option to visualize periodic replicas such as a `2 x 2 x 1` display.
- Add a display-only conventional-cell conversion before visualization.
- Label the view with the explicit POSCAR atom count and whether the displayed
  cell is primitive, conventional, or replicated.
- Do not silently replace the calculation input structure with a supercell just
  to improve the picture; visualization replication and structure modification
  should be separate operations.

