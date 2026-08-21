import json
import subprocess
from typing import Dict, Any


UMA_PYTHON = r"C:\Users\iAI\Desktop\VASPilot\uma_env\Scripts\python.exe"
UMA_RUNNER = r"C:\Users\iAI\Desktop\VASPilot\uma_runner.py"


def uma_relaxation(structure_path: str) -> Dict[str, Any]:
    """
    Relax a structure using the external UMA environment.

    Args:
        structure_path: Path to the input structure file (e.g., CIF, POSCAR).

    Returns:
        The parsed JSON result returned by uma_runner.py.
    """
    try:
        process = subprocess.run(
            [UMA_PYTHON, UMA_RUNNER, structure_path],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"success": False, "error": f"Failed to start UMA subprocess: {exc}"}

    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        error = f"UMA runner returned invalid JSON: {exc}"
        if process.stderr.strip():
            error += f"; stderr: {process.stderr.strip()}"
        return {"success": False, "error": error}

    if not isinstance(result, dict):
        return {"success": False, "error": "UMA runner JSON result is not an object"}

    if process.returncode != 0 and result.get("success") is not False:
        return {
            "success": False,
            "error": process.stderr.strip()
            or f"UMA subprocess exited with code {process.returncode}",
        }

    return result
