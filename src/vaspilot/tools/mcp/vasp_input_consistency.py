"""Read and compare the VASP inputs that define an energy calculation."""

import hashlib
import json
import re
from pathlib import Path

from pymatgen.io.vasp import Incar, Kpoints


# Ionic, output, restart, and execution controls are selected by the SCF stage.
# All other explicit source tags carry through, including less common XC and
# vdW settings that cannot safely be enumerated in an allowlist.
SCF_STAGE_TAGS = frozenset({
    "SYSTEM", "NSW", "IBRION", "ISIF", "EDIFFG", "POTIM", "ICHARG",
    "ISTART", "LWAVE", "LCHARG", "LAECHG", "LORBIT", "NCORE",
    "NPAR", "KPAR", "NELM", "EDIFF", "AMIN", "NWRITE", "NSIM",
    "LPLANE", "LSCALU",
})


def read_restart_inputs(directory):
    """Require a completed calculation's physical inputs before restarting it."""
    directory = Path(directory)
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Restart source is missing a nonempty {name}: {path}")
    try:
        return Incar.from_file(directory / "INCAR"), Kpoints.from_file(directory / "KPOINTS"), directory / "POTCAR"
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read restart VASP inputs in {directory}: {exc}") from exc


def compact_setting(value):
    if isinstance(value, list) and len(value) > 32:
        digest = hashlib.sha256(json.dumps(value, default=str).encode()).hexdigest()
        return {"count": len(value), "sha256": digest}
    return value


def input_provenance(directory):
    """Capture effective input identity from files actually written for a job."""
    directory = Path(directory)
    incar, kpoints, potcar_path = read_restart_inputs(directory)
    potcar = potcar_path.read_bytes()
    titles = [title.decode("utf-8", errors="replace").strip() for title in
              re.findall(rb"^\s*TITEL\s*=\s*(.*?)\s*$", potcar, re.MULTILINE)]
    enmax = [float(value) for value in re.findall(rb"\bENMAX\s*=\s*([0-9.]+)", potcar)]
    if not titles or not enmax:
        raise ValueError(f"POTCAR has no TITEL or ENMAX entries: {potcar_path}")
    mesh = None
    if kpoints.num_kpts == 0 and kpoints.kpts:
        mesh = [int(n) for n in kpoints.kpts[0]]
    return {
        "potcar_sha256": hashlib.sha256(potcar).hexdigest(),
        "potcar_titles": titles,
        "encut_eV": float(incar["ENCUT"]) if "ENCUT" in incar else max(enmax),
        "kpoint_mesh": mesh,
        "kpoints_sha256": hashlib.sha256((directory / "KPOINTS").read_bytes()).hexdigest(),
        "incar_sha256": hashlib.sha256((directory / "INCAR").read_bytes()).hexdigest(),
        "electronic_settings": {key: compact_setting(value) for key, value in incar.items()
                                if key not in SCF_STAGE_TAGS},
        "restart_policy": {key: int(incar[key]) for key in ("ICHARG", "ISTART") if key in incar},
    }
