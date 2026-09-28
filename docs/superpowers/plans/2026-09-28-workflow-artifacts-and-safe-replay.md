# Workflow Artifacts and Safe Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VASPilot's artifact handoffs type-checked and exact repeated actions safe across candidate generation, UMA relaxation, VASP relaxation, SCF, and Bader.

**Architecture:** Add a small artifact resolver and a SQLite action ledger beside the existing calculation database. Each migrated MCP action validates its input, claims a deterministic action key before external work, and returns an existing result for a repeated request. Keep legacy tool arguments and add compact typed references to outputs; expose one read-only inspection tool for handoffs.

**Tech Stack:** Python 3.12, FastMCP, CrewAI, SQLite, pytest, existing JSON manifests and SLURM wrappers.

**Spec:** [Workflow Artifacts and Safe Replay Design](../specs/2026-09-28-workflow-artifacts-and-replay-design.md)

## Global Constraints

- Preserve current MCP tool names and existing path/calculation-ID arguments; `artifact_ref` is additive.
- Do not run real VASP, UMA, or SLURM in automated tests. Mock only the external subprocess/submission boundary.
- A wrong artifact kind must fail before any subprocess or `sbatch` call.
- Do not alter or delete existing manifests, calculation records, or active jobs during migration.
- A changed scientific parameter set gets a different action key; for UMA and later stages, the same source/stage returns `STAGE_CONFLICT` unless an explicit branch mechanism is added later.
- First rollout covers candidate generation → UMA relaxation → VASP relaxation → SCF → Bader. Other tools remain unchanged.

## Review Focus

- A relaxation-result manifest passed as `candidate_manifest_path` must return `INVALID_ARTIFACT_KIND`, expected/actual kinds, and `next_tool=vasp_relaxation`, with no subprocess call (Task 1/3).
- Two concurrent identical UMA requests must launch one runner; the other receives `ACTION_IN_PROGRESS` or the completed artifact (Task 2/3).
- A server crash after `sbatch` but before ledger receipt must never trigger automatic resubmission (Task 2/4).
- A completed calculation whose SLURM history expired must still resolve as completed (Task 4/6).
- A manifest changed after a typed ref was issued must return `ARTIFACT_CHANGED` rather than silently supplying different candidate data (Task 1/6).

---

## File map

| File | Responsibility |
| --- | --- |
| `src/vaspilot/tools/mcp/workflow_artifacts.py` | Typed references, manifest/calculation resolution, compact structured errors. |
| `src/vaspilot/tools/mcp/workflow_actions.py` | SQLite action keys, atomic claims, stored receipts, stale-claim reconciliation state. |
| `src/vaspilot/tools/mcp/mcp_server.py` | MCP adapters: validate, claim, submit, return refs; read-only inspection tool. |
| `src/vaspilot/tools/mcp/adsorption_tools.py` | Keep subprocess runners; accept validated inputs from MCP wrapper. |
| `src/vaspilot/scripts/adsorption_relaxation_runner.py` | Keep direct CLI validation aligned with artifact-kind contract. |
| `src/vaspilot/tools/mcp/sqlite_database.py` | Remains scientific calculation storage; do not embed action-ledger policy in it. |
| `examples/1.Basic/configs/crew_config_en.yaml` | Tell agents to carry exact refs and inspect read-only. |
| `tests/test_workflow_artifacts.py`, `tests/test_workflow_actions.py`, `tests/test_workflow_handoffs.py` | Contract, replay/concurrency, and cross-stage tests. |

### Task 1: Typed references and validation

**Files:** Create `src/vaspilot/tools/mcp/workflow_artifacts.py`; create `tests/test_workflow_artifacts.py`.

**Interfaces:** Produce `make_manifest_ref(manifest: dict, path: Path) -> dict`, `resolve_manifest(ref_or_path: dict | str, expected_kind: str) -> tuple[dict, dict]`, and `resolve_calculation(ref_or_id: dict | str, expected_type: str, read_record: Callable) -> tuple[dict, dict]`. Raise `ArtifactInputError` with `to_result()` for MCP errors.

- [ ] **Step 1: Write failing tests.** Use a real temporary JSON manifest; assert a matching ref resolves, a modified manifest raises `ARTIFACT_CHANGED`, and a `ranked_adsorption_relaxations` manifest supplied as `adsorption_candidate_set` raises `INVALID_ARTIFACT_KIND` with `next_tool="vasp_relaxation"`. Assert a Bader record is rejected when SCF is expected.

```python
candidate_manifest = {"success": True, "artifact_kind": "adsorption_candidate_set",
                      "artifact_id": "set-1", "candidates": []}
path = tmp_path / "candidate_manifest.json"
path.write_text(json.dumps(candidate_manifest))
ref = make_manifest_ref(candidate_manifest, path)
assert resolve_manifest(ref, "adsorption_candidate_set")[0]["artifact_id"] == "set-1"
path.write_text(json.dumps({**candidate_manifest, "artifact_id": "set-2"}))
with pytest.raises(ArtifactInputError) as error:
    resolve_manifest(ref, "adsorption_candidate_set")
assert error.value.code == "ARTIFACT_CHANGED"
```

- [ ] **Step 2: Run** `python -m pytest tests/test_workflow_artifacts.py -q`; confirm it fails because the module is absent.
- [ ] **Step 3: Implement** a frozen typed-ref schema using plain JSON dictionaries, SHA-256 of manifest bytes, JSON shape/kind/ID checks, and a calculation resolver that reads the existing DB callback. Use `Path.resolve()` for a legacy path and validate its content. Keep errors as data at the MCP boundary.

```python
class ArtifactInputError(ValueError):
    def __init__(self, code: str, message: str, *, expected_kind=None,
                 actual_kind=None, next_tool=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.expected_kind, self.actual_kind = expected_kind, actual_kind
        self.next_tool = next_tool

    def to_result(self) -> dict:
        return {"success": False, "error_code": self.code,
                "error": self.message, "retryable": False,
                "expected_kind": self.expected_kind,
                "actual_kind": self.actual_kind, "next_tool": self.next_tool}
```

- [ ] **Step 4: Run** `python -m pytest tests/test_workflow_artifacts.py -q`; expect all new tests to pass. Commit only these two files.

### Task 2: Atomic action ledger

**Files:** Create `src/vaspilot/tools/mcp/workflow_actions.py`; create `tests/test_workflow_actions.py`.

**Interfaces:** Produce `action_key(tool: str, version: int, source: dict, params: dict) -> str`, `stage_key(tool: str, source: dict, selection: dict) -> str`, and `WorkflowActionStore(db_path: str)` with `claim(key, tool, stage_key=None) -> Claim`, `record_submission(key, response)`, `finish(key, response)`, `get(key)`, and `mark_stale_for_reconciliation(key)`. `Claim.acquired` distinguishes the one caller allowed to start external work.

- [ ] **Step 1: Write failing tests.** Assert canonical parameter ordering yields one key; changing `fmax` changes it; two separate SQLite connections claiming the same key yield one winner; a different `fmax` under the same UMA stage key returns `STAGE_CONFLICT`; submitted/completed responses are reusable; failed/cancelled responses do not auto-retry; a stale claim becomes `reconciliation_required` and cannot be claimed again.

```python
key = action_key("relax_adsorption_candidates", 1,
                 {"kind": "adsorption_candidate_set", "id": "set-1"},
                 {"fmax": 0.01, "max_steps": 200})
stage = stage_key("relax_adsorption_candidates",
                  {"kind": "adsorption_candidate_set", "id": "set-1"}, {})
first = WorkflowActionStore(str(db_path)).claim(key, "relax_adsorption_candidates", stage)
second = WorkflowActionStore(str(db_path)).claim(key, "relax_adsorption_candidates", stage)
assert first.acquired is True and second.acquired is False
```

- [ ] **Step 2: Run** `python -m pytest tests/test_workflow_actions.py -q`; confirm import/API failures.
- [ ] **Step 3: Implement** a `workflow_actions` table with `key TEXT PRIMARY KEY`, nullable unique `stage_key`, `tool`, `state`, `response_json`, `calculation_id`, `created_at`, and `updated_at`. Use `BEGIN IMMEDIATE` for claims. Store `preparing`, `submitted`, `completed`, `failed`, `cancelled`, and `reconciliation_required`. For a second caller during `preparing`, return `ACTION_IN_PROGRESS` and the action key; never execute the external work. Do not infer that a stale `preparing` row is safe to rerun.

```sql
CREATE TABLE IF NOT EXISTS workflow_actions (
  key TEXT PRIMARY KEY, stage_key TEXT UNIQUE,
  tool TEXT NOT NULL, state TEXT NOT NULL,
  response_json TEXT, calculation_id TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- [ ] **Step 4: Run** `python -m pytest tests/test_workflow_actions.py -q`; expect pass. Commit the ledger and tests.

### Task 3: Candidate generation and UMA relaxation adapters

**Files:** Modify `src/vaspilot/tools/mcp/mcp_server.py`, `src/vaspilot/scripts/adsorption_relaxation_runner.py`; create `tests/test_workflow_handoffs.py`.

**Interfaces:** Candidate generation returns its existing result plus `artifact_ref`. `relax_adsorption_candidates(candidate_manifest_path=...)` accepts legacy path, resolves kind `adsorption_candidate_set`, then claims an action key before invoking the existing subprocess. It returns `ranked_adsorption_relaxations` ref or a structured error. Existing direct candidate-list mode stays available.

- [ ] **Step 1: Write failing MCP tests.** Mock the candidate/UMA subprocess functions, not validation or the action store. Verify wrong-kind input invokes neither runner; two identical UMA calls invoke the runner once and return the same manifest; a different `fmax` for the same candidate set returns `STAGE_CONFLICT` without invoking the runner. Use temporary manifests with literal kinds and IDs.

```python
wrong = tmp_path / "relaxation_manifest.json"
wrong.write_text(json.dumps({"success": True,
    "artifact_kind": "ranked_adsorption_relaxations", "artifact_id": "relax-1"}))
result = asyncio.run(tools["relax_adsorption_candidates"](
    candidate_manifest_path=str(wrong)))
assert result["error_code"] == "INVALID_ARTIFACT_KIND"
assert result["next_tool"] == "vasp_relaxation"
assert runner_calls == []
```

- [ ] **Step 2: Run** `python -m pytest tests/test_workflow_handoffs.py -q`; confirm the new assertions fail.
- [ ] **Step 3: Implement** resolver + ledger calls in the two MCP adapters. Add `artifact_ref` to successful responses without removing existing fields. In `adsorption_relaxation_runner.validate_request`, check `artifact_kind` before looking for `candidates`, so direct CLI calls get the same specific error. On subprocess exception or malformed JSON, finish the action as failed; do not leave it `preparing` except for an actual process crash.
- [ ] **Step 4: Run** `python -m pytest tests/test_workflow_handoffs.py tests/test_adsorption_candidate_runner.py tests/test_adsorption_relaxation_runner.py -q`; account for the repository's existing missing-`fairchem` failures separately. Commit the migrated adapters and tests.

### Task 4: VASP relaxation, SCF, and Bader submission replay

**Files:** Modify `src/vaspilot/tools/mcp/mcp_server.py`, `src/vaspilot/tools/mcp/bader_analysis.py`; extend `tests/test_workflow_handoffs.py`.

**Interfaces:** `vasp_relaxation`, `vasp_scf`, and `run_bader` keep current arguments. Their successful result adds a calculation `artifact_ref`. Each validates its source ref or legacy ID/path, claims an action key, submits once, and records the returned calculation ID before responding.

- [ ] **Step 1: Write failing tests.** With fake VASP/SLURM submitters, verify exact repeated calls return one SLURM ID, an active response returns the same calculation ID, a wrong source type submits nothing, and a completed SCF from the DB remains complete despite a fake expired SLURM lookup. Add a test that `run_bader` refuses an SCF missing AECCAR0/AECCAR2 before ledger claim.

```python
first = asyncio.run(tools["vasp_scf"](
    restart_id="relax-1", analyses=["bader"]))
second = asyncio.run(tools["vasp_scf"](
    restart_id="relax-1", analyses=["bader"]))
assert first["calculation_id"] == second["calculation_id"]
assert second["reused"] is True
assert len(slurm_submissions) == 1
```

- [ ] **Step 2: Run** `python -m pytest tests/test_workflow_handoffs.py -q`; confirm repeated submission still occurs.
- [ ] **Step 3: Implement** validation and ledger claim at each MCP submission entry point before UUID generation or `sbatch`. Keep the calculation database as the source for live status and final results. Use the action-ledger row as the durable submission receipt so an interrupted submission can be reconciled; if a claim is stale with no durable receipt, return `ACTION_RECONCILIATION_REQUIRED` rather than submit again. Record `submitted` immediately when the SLURM ID arrives.
- [ ] **Step 4: Run** `python -m pytest tests/test_workflow_handoffs.py tests/test_adsorption_scf_mcp.py tests/test_bader_submission.py tests/test_relaxation_manifest_mcp.py -q`; expect migrated-path tests to pass. Commit the submission adapters and tests.

### Task 5: Read-only inspection and agent handoff

**Files:** Modify `src/vaspilot/tools/mcp/mcp_server.py`, `examples/1.Basic/configs/crew_config_en.yaml`; extend `tests/test_workflow_handoffs.py`.

**Interfaces:** Register `inspect_workflow_artifact(artifact_ref: dict) -> dict`. Return compact manifest/calculation summaries and exact successor inputs, including eligible candidate IDs. No computation, DB mutation, or SLURM query. The agent instructions pass refs across delegation and use inspection for verification.

- [ ] **Step 1: Write failing tests.** Feed a four-candidate UMA manifest and assert the inspection response includes its exact `manifest_path`, `best_candidate_id`, and candidate IDs while excluding trajectories and large per-step data. Spy on subprocess/SLURM functions and assert zero calls.

```python
summary = asyncio.run(tools["inspect_workflow_artifact"](artifact_ref=ref))
assert summary["kind"] == "ranked_adsorption_relaxations"
assert summary["candidate_ids"] == ["candidate_000", "candidate_001",
                                     "candidate_002", "candidate_003"]
assert external_calls == []
```

- [ ] **Step 2: Run** `python -m pytest tests/test_workflow_handoffs.py -q`; confirm the inspection tool is absent.
- [ ] **Step 3: Implement** the read-only MCP tool through the shared resolver. Update the vasp agent's tool list and prompt: preserve `artifact_ref` through delegate handoffs; inspect with `inspect_workflow_artifact`; pass the returned manifest to `vasp_relaxation`; stop on non-retryable errors. Avoid promising that prose alone enforces the rule.
- [ ] **Step 4: Run** `python -m pytest tests/test_workflow_handoffs.py -q` and parse the YAML with `python -c 'import yaml; yaml.safe_load(open("examples/1.Basic/configs/crew_config_en.yaml"))'`. Commit the inspection tool, configuration, and tests.

### Task 6: Cross-stage acceptance and migration notes

**Files:** Extend `tests/test_workflow_handoffs.py`; create `docs/workflow-artifact-contract.md`.

**Interfaces:** Document `artifact_ref` shape, action-key semantics, structured errors, compatibility, and the steps required to migrate one more MCP action. Keep the implementation spec linked.

- [ ] **Step 1: Write a failing end-to-end fake-runner test** for generation → UMA → VASP relaxation → Bader-configured SCF → Bader, including a repeated UMA call and an expired SLURM-history stub. Assert each external action runs once and the final Bader record still points to its SCF ID. Add the changed-manifest-digest case.
- [ ] **Step 2: Run** `python -m pytest tests/test_workflow_handoffs.py -q`; confirm the integration test catches the remaining gap.
- [ ] **Step 3: Complete** the smallest integration fixes needed, then document a concrete next-tool migration checklist with resolver, key inputs, ledger state, and terminal result mapping. Name adsorption-energy and NSCF as later migrations; do not silently convert them in this patch.
- [ ] **Step 4: Run** the focused tests, `python -m pytest -q --tb=short`, and `git diff --check`. Report all failures by name; the current repository baseline has two tests failing without `fairchem` and one surface test expecting `orthogonalize_c=False`. Commit the final integration test and docs after review.

## Handoff notes

Before implementation, inspect the working tree: this repository currently has uncommitted Bader/SCF work and untracked tests. Do not overwrite it or assume a clean baseline. Use an isolated worktree only after confirming it contains the needed uncommitted changes, or explicitly carry those changes forward. Do not cancel active SLURM jobs or rewrite scientific records as part of this patch.

At completion, demonstrate one fake-runner trace where a delegated agent makes the same wrong second UMA call: the server returns the typed error and zero additional runner calls. Demonstrate a second trace where the agent repeats a correct request: it receives the prior manifest/calculation ID with `reused=true`. State clearly that these server guarantees do not eliminate every bad *attempted* tool call from a free-text agent.
