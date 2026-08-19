# Deterministic structure-retrieval generalization benchmark

- Generated: 2026-08-18T08:06:15.988520+00:00
- Runs per prompt: 5
- Total runs: 80
- Failures: 0
- Fully consistent prompts: 16/16
- MP_API_KEY leak detected: False

| Case | Classification | Resolver | Selected ID | Consistent | Retries | Failures | Crew starts | Files |
|---|---|---|---|---:|---:|---:|---|---|
| A | {"pure_mp_structure": 5} | {"search_results": 5} | {"null": 5} | yes | 0 | 0 | {"0": 5} | {"0": 5} |
| B | {"pure_mp_structure": 5} | {"selected": 5} | {"mp-1434": 5} | yes | 0 | 0 | {"0": 5} | {"1": 5} |
| C | {"pure_mp_structure": 5} | {"search_results": 5} | {"null": 5} | yes | 0 | 0 | {"0": 5} | {"0": 5} |
| D | {"pure_mp_structure": 5} | {"selected": 5} | {"mp-2815": 5} | yes | 0 | 0 | {"0": 5} | {"1": 5} |
| E | {"pure_mp_structure": 5} | {"search_results": 5} | {"null": 5} | yes | 5 | 0 | {"0": 5} | {"0": 5} |
| F | {"mixed_mp_structure": 5} | {"selected": 5} | {"mp-224": 5} | yes | 0 | 0 | {"1": 5} | {"1": 5} |
| G | {"mixed_mp_structure": 5} | {"selected": 5} | {"mp-2657": 5} | yes | 0 | 0 | {"1": 5} | {"1": 5} |
| H | {"mixed_mp_structure": 5} | {"selected": 5} | {"mp-149": 5} | yes | 0 | 0 | {"1": 5} | {"1": 5} |
| I | {"mixed_mp_structure": 5} | {"selected": 5} | {"mp-2815": 5} | yes | 0 | 0 | {"1": 5} | {"1": 5} |
| J | {"mixed_mp_structure": 5} | {"ambiguous": 5} | {"null": 5} | yes | 0 | 0 | {"0": 5} | {"0": 5} |
| K | {"mixed_mp_structure": 5} | {"ambiguous": 5} | {"null": 5} | yes | 0 | 0 | {"0": 5} | {"0": 5} |
| L | {"clarification_required": 5} | {"null": 5} | {"null": 5} | yes | 0 | 0 | {"0": 5} | {"0": 5} |
| M | {"clarification_required": 5} | {"null": 5} | {"null": 5} | yes | 0 | 0 | {"0": 5} | {"0": 5} |
| N | {"local_or_unrelated": 5} | {"null": 5} | {"null": 5} | yes | 0 | 0 | {"1": 5} | {"0": 5} |
| O | {"local_or_unrelated": 5} | {"null": 5} | {"null": 5} | yes | 0 | 0 | {"1": 5} | {"0": 5} |
| P | {"local_or_unrelated": 5} | {"null": 5} | {"null": 5} | yes | 0 | 0 | {"1": 5} | {"0": 5} |

See the JSON report for requests, ordered IDs, diagnostics, and per-run timings.
