from __future__ import annotations

import unittest

from vaspilot.crew.vasp_crew import VaspCrew


class VaspCrewStructurePolicyTests(unittest.TestCase):
    def test_stale_yaml_cannot_restore_forbidden_structure_tools(self):
        crew = object.__new__(VaspCrew)
        crew.config = {
            "agents": {
                "crystal": {
                    "tools": [
                        "search_materials_project",
                        "analyze_crystal_structure",
                        "create_crystal_structure",
                    ]
                }
            }
        }
        allowed = crew._allowed_tool_names(
            "crystal",
            frozenset({"search_materials_project", "create_crystal_structure"}),
        )
        self.assertEqual(allowed, ["analyze_crystal_structure"])

    def test_unrelated_tasks_retain_configured_tools(self):
        crew = object.__new__(VaspCrew)
        crew.config = {
            "agents": {
                "crystal": {
                    "tools": ["search_materials_project", "create_crystal_structure"]
                }
            }
        }
        self.assertEqual(
            crew._allowed_tool_names("crystal"),
            ["search_materials_project", "create_crystal_structure"],
        )


if __name__ == "__main__":
    unittest.main()
