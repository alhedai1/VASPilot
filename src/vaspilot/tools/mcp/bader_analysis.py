"""Run Bader analysis on the charge files of a completed VASP SCF job."""

import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from pymatgen.io.vasp import Chgcar


def submit_bader_job(
    scf_calculation_id: str,
    scf_record: dict | None,
    work_dir: Path,
    bader_executable: str | None,
    *,
    python_executable: str | None = None,
    partition: str | None = None,
) -> dict:
    """Submit a Bader post-processing job for one completed SCF record."""
    def invalid(message: str) -> dict:
        return {
            "success": False,
            "status": "failed",
            "error_code": "INVALID_BADER_INPUT",
            "error": message,
            "retryable": False,
            "calculation_id": None,
            "slurm_id": None,
        }

    if not scf_record or scf_record.get("calc_type") != "scf":
        return invalid(f"Calculation {scf_calculation_id} is not an SCF record")
    if scf_record.get("status") != "completed":
        return invalid(f"SCF calculation {scf_calculation_id} must be completed")
    if not scf_record.get("calculate_path"):
        return invalid("SCF record has no calculate_path")
    scf_dir = Path(scf_record["calculate_path"]).resolve()
    for name in ("CHGCAR", "AECCAR0", "AECCAR2"):
        if not (scf_dir / name).is_file():
            return invalid(f"Required SCF output is missing: {scf_dir / name}")
    if not bader_executable:
        return invalid("Set bader_executable in the MCP configuration")
    binary = shutil.which(bader_executable)
    if binary is None:
        return invalid(f"Bader executable is unavailable: {bader_executable}")
    if partition is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", partition):
        return invalid("bader_partition contains invalid characters")

    calculation_id = str(uuid.uuid4())
    output = Path(work_dir).resolve() / calculation_id / "bader"
    output.mkdir(parents=True, exist_ok=False)
    command = [
        python_executable or sys.executable, "-m", "vaspilot.tools.mcp.bader_analysis",
        str(scf_dir), str(output), binary,
    ]
    script = output / "bader.sbatch"
    script.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=vaspilot-bader\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=16G\n"
        "#SBATCH --time=02:00:00\n"
        "#SBATCH --output=bader.log\n"
        "#SBATCH --error=bader.err\n"
        "set -euo pipefail\n"
        f"{shlex.join(command)}\n"
    )
    submission = ["sbatch", "--parsable"]
    if partition:
        submission.append(f"--partition={partition}")
    submission.append(str(script))
    try:
        process = subprocess.run(submission, cwd=output, capture_output=True, text=True)
    except OSError as exc:
        return invalid(f"Could not start sbatch: {exc}")
    job_id = process.stdout.strip().split(";", 1)[0]
    if process.returncode or not job_id.isdigit():
        return invalid(f"Bader submission failed: {process.stderr.strip() or process.stdout.strip()}")
    return {
        "success": True,
        "error": None,
        "status": "submitted",
        "calc_type": "bader",
        "calculation_id": calculation_id,
        "slurm_id": job_id,
        "calculate_path": str(output),
        "restart_id": scf_calculation_id,
    }


def parse_acf(path: Path, expected_atoms: int) -> list[dict]:
    """Read the one-based atom indices and integrated electrons from ACF.dat."""
    atoms = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 7 or not fields[0].isdigit():
            continue
        index = int(fields[0])
        electrons = float(fields[4])
        if index != len(atoms) + 1 or not math.isfinite(electrons):
            raise ValueError("ACF.dat contains invalid atom indices or charges")
        atoms.append({"index": index, "electrons": electrons})
    if len(atoms) != expected_atoms:
        raise ValueError(
            f"ACF.dat atom count {len(atoms)} does not match structure atom count {expected_atoms}"
        )
    return atoms


def run_bader_analysis(scf_dir: Path, output_dir: Path, bader_executable: str) -> dict:
    """Sum the all-electron density, run Bader, and save parsed results."""
    scf_dir = Path(scf_dir).resolve()
    output_dir = Path(output_dir).resolve()
    for name in ("CHGCAR", "AECCAR0", "AECCAR2"):
        if not (scf_dir / name).is_file():
            raise FileNotFoundError(f"Required SCF output is missing: {scf_dir / name}")

    core = Chgcar.from_file(scf_dir / "AECCAR0")
    valence = Chgcar.from_file(scf_dir / "AECCAR2")
    if core.structure != valence.structure or core.dim != valence.dim:
        raise ValueError("AECCAR0 and AECCAR2 have different structures or grids")
    output_dir.mkdir(parents=True, exist_ok=True)
    (core + valence).write_file(output_dir / "CHGCAR_sum")

    process = subprocess.run(
        [bader_executable, str(scf_dir / "CHGCAR"), "-ref", "CHGCAR_sum"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    (output_dir / "bader.stdout").write_text(process.stdout)
    (output_dir / "bader.stderr").write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f"Bader exited with code {process.returncode}: {process.stderr[-1000:]}")
    acf = output_dir / "ACF.dat"
    if not acf.is_file():
        raise FileNotFoundError("Bader exited successfully but did not write ACF.dat")
    atoms = parse_acf(acf, len(core.structure))
    result = {
        "success": True,
        "source_scf_directory": str(scf_dir),
        "acf_path": str(acf),
        "atoms": atoms,
    }
    (output_dir / "bader_result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bader analysis on completed VASP SCF outputs")
    parser.add_argument("scf_dir")
    parser.add_argument("output_dir")
    parser.add_argument("bader_executable")
    args = parser.parse_args()
    run_bader_analysis(Path(args.scf_dir), Path(args.output_dir), args.bader_executable)


if __name__ == "__main__":
    main()
