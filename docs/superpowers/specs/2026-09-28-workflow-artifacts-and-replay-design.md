# Workflow Artifacts and Safe Replay Design

## Purpose

An agent repeated `relax_adsorption_candidates` to inspect a successful UMA result. It passed a `ranked_adsorption_relaxations` manifest to an input that expected an `adsorption_candidate_set` manifest. The second call failed before computation, but the agent then changed course based on a misleading error. Similar confusion can occur at relaxation → SCF and SCF → Bader boundaries, and repeated submissions can waste cluster time.

## Scope

Build reusable artifact validation and action replay controls, then apply them to the candidate generation → UMA relaxation → VASP relaxation → SCF → Bader path. Preserve existing MCP tool names and legacy path/ID arguments during migration. Other tools keep their current behavior until separately migrated; the shared components must be reusable by them.

This patch protects tool execution. CrewAI's free-text delegation may still propose a wrong call; the server must reject a wrong artifact kind, return a useful next action, and prevent exact duplicate work. It does not replace CrewAI with a deterministic workflow engine.

## Artifact contract

Every migrated successful tool result includes an `artifact_ref` object with `kind`, `id`, and exactly one of `manifest_path` or `calculation_id`. Manifest refs also include `sha256` of the manifest bytes. Calculation kinds are `vasp_relaxation_calculation`, `vasp_scf_calculation`, and `bader_calculation`; these map to the existing database `calc_type` values `relaxation`, `scf`, and `bader`. The existing top-level `manifest_path` or `calculation_id` remains for compatibility. The tool's existing manifest `artifact_kind` and `artifact_id` remain authoritative. Calculation refs use the saved database `calc_type` and `calculation_id`.

The shared resolver reads a manifest once, checks JSON shape, success state, kind, ID, and optional digest, then returns the validated payload. For calculation refs it reads the calculation DB and checks type and required state. A mismatch returns a structured, non-retryable error with `error_code=INVALID_ARTIFACT_KIND`, `expected_kind`, `actual_kind`, and `next_tool` where known. A missing or changed artifact gets a distinct code. No compute or SLURM submission occurs before validation.

## Read-only inspection

Add `inspect_workflow_artifact(artifact_ref)`. For candidate and UMA manifests it returns compact counts, exact manifest path, and eligible candidate IDs/paths; for VASP and Bader calculation refs it returns stored status and existing result locations. It performs no subprocess or SLURM action. Agents can use it to verify a handoff rather than repeat a producer tool. Results remain bounded so large candidate lists do not bury the reference.

## Safe replay

An action key is SHA-256 of canonical JSON containing tool name, implementation version, validated source artifact kind/ID/digest (or input file digests for generation), and normalized scientific parameters. Do not include output-directory path, timestamps, or conversation ID. A SQLite `workflow_actions` table with a unique key makes claims atomic across Quart tasks and MCP requests.

For an identical request, return the existing completed artifact or submitted/running calculation ID with `reused=true`; never launch a second subprocess or `sbatch`. A previously failed/cancelled action returns its terminal result. For UMA relaxation and downstream actions, also reserve a `stage_key` derived from tool name, source artifact, candidate ID where relevant, and analysis intent, excluding scientific settings. A second request for that stage with changed settings returns `STAGE_CONFLICT` and the existing action/result; it does not launch work. Candidate generation uses exact-call deduplication only because varying placement parameters is a normal exploration step. Intentional variants or retries need a separate user-directed branch mechanism, outside this first patch. If a process dies in the narrow interval between external submission and receipt persistence, mark the action `reconciliation_required` and refuse automatic resubmission; the job identity must be reconciled first.

The action ledger is separate from scientific result records. It stores a compact response and links to existing manifests/calculation IDs; it never copies CHGCAR, WAVECAR, or full result blobs. Completed scientific records remain authoritative after SLURM history expires.

## Stage contracts

| Consumer | Required source | Required state | Output |
| --- | --- | --- | --- |
| `relax_adsorption_candidates` | `adsorption_candidate_set` | successful manifest | `ranked_adsorption_relaxations` ref |
| `vasp_relaxation` (manifest mode) | `ranked_adsorption_relaxations` + eligible candidate ID | converged candidate | VASP relaxation calculation ref |
| `vasp_scf` (restart mode) | VASP relaxation calculation ID | completed | VASP SCF calculation ref |
| `vasp_scf` (UMA manifest mode) | `ranked_adsorption_relaxations` | eligible candidate | VASP SCF calculation ref |
| `run_bader` | VASP SCF calculation ID | completed and charge files present | Bader calculation ref |

Keep direct structure-path modes available for users who intentionally bypass the artifact chain. Bader intent on `vasp_scf` continues to set `NSW=0`, `IBRION=-1`, `LCHARG=True`, and `LAECHG=True`.

## Compatibility and limits

Existing manifests are accepted by their legacy path arguments after validation; no bulk rewrite is required. Existing calculation IDs remain valid. The new `artifact_ref` is additive. Database migration adds only the action ledger. In-flight jobs started before the patch remain monitorable but are not retroactively deduplicated without a known action key. The first rollout does not migrate adsorption-energy, NSCF, or other structure tools; these are later candidates for the same resolver and ledger.

## Acceptance

A wrong manifest kind fails before subprocess/SLURM with an actionable error. A second exact UMA request returns the first manifest. A second UMA request for the same candidate set with a different `fmax` returns `STAGE_CONFLICT`. Two concurrent identical requests launch one action. Repeated VASP/SCF/Bader submissions return the same calculation ID. A completed SCF remains completed when SLURM has forgotten its ID. An agent can inspect the successful UMA result without invoking relaxation again. Legacy path and calculation-ID calls continue to work. No test requires real VASP or a live SLURM allocation.
