from vaspilot.tools.mcp import adsorption_workflow


def test_workflow_passes_artifacts_between_stages(monkeypatch):
    monkeypatch.setattr(
        adsorption_workflow,
        "retrieve_bulk_parent",
        lambda **kwargs: {"success": True, "structure_path": "/bulk.vasp"},
    )
    monkeypatch.setattr(
        adsorption_workflow,
        "build_surface",
        lambda path, miller: {
            "success": True,
            "slab_structure_path": "/slab.vasp",
            "num_atoms": 80,
        },
    )
    monkeypatch.setattr(
        adsorption_workflow,
        "build_adsorbate",
        lambda name, directory: {
            "success": True,
            "adsorbate_structure_path": "/CO.xyz",
            "binding_atom_indices": [1],
            "num_atoms": 2,
        },
    )
    monkeypatch.setattr(
        adsorption_workflow,
        "generate_adsorption_candidates",
        lambda **kwargs: {
            "success": True,
            "candidates": [
                {"structure_path": "/candidate_000.vasp", "num_atoms": 82},
                {"structure_path": "/candidate_001.vasp", "num_atoms": 82},
            ],
        },
    )

    captured = {}

    def fake_relax(**kwargs):
        captured.update(kwargs)
        return {"success": True, "results": []}

    monkeypatch.setattr(
        adsorption_workflow, "relax_adsorption_candidates", fake_relax
    )

    result = adsorption_workflow.run_adsorption_workflow(
        api_key="key",
        download_path="/downloads",
        material_id="mp-126",
        miller_index=[1, 1, 1],
        adsorbate="CO",
    )

    assert result["success"] is True
    assert result["expected_atoms_per_candidate"] == 82
    assert captured["candidate_structure_paths"] == [
        "/candidate_000.vasp",
        "/candidate_001.vasp",
    ]
    assert captured["fmax"] == 0.05
    assert captured["max_steps"] == 200
