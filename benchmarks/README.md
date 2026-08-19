# Structure-retrieval benchmarks

Run the live, no-VASP generalization benchmark from the repository root:

```powershell
.\.venv\Scripts\python.exe benchmarks\structure_retrieval_generalization.py
```

The default is five independent executions per prompt. Use `--runs N` to
change it and `--output-dir PATH` to choose where JSON and Markdown reports
are written. The benchmark requires `MP_API_KEY` in the process environment
or the repository's untracked `.env` file. It uses live configured LLM and
Materials Project access, so it is intentionally not part of the unit tests.

Downstream Crew execution is replaced by a spy and VASP is never run.
