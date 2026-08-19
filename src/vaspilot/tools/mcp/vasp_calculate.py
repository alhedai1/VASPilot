import os
import time
import subprocess
import shutil
from pymatgen.core import Element, Structure
from pymatgen.io.vasp import VaspInput, Vasprun, Kpoints, Poscar, Potcar
from typing import Optional, Dict, Any
import numpy as np


def _submit_slurm_job(calc_type: str, calculate_path: str, 
                     attachment_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generic SLURM job submission method

    Args:
        calc_type: Calculation type ("relaxation", "scf", "nscf")
        calculate_path: Calculation directory path
        attachment_path: Attachment path containing the SLURM script and other files

    Returns:
        Dict containing slurm_id, calc_type, calculate_path, success, error, status, etc.
    """
    try:
        # If an attachment path is given, copy the attachment files into the calculation directory
        if attachment_path is not None and os.path.exists(attachment_path):
            # Copy all files under the attachment directory into the calculation directory
            for file_name in os.listdir(attachment_path):
                src_file = os.path.join(attachment_path, file_name)
                dst_file = os.path.join(calculate_path, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)

        # Locate the SLURM script file
        slurm_script_path = None
        for script_name in ['submit.sh', 'run.sh', 'slurm.sh']:
            script_path = os.path.join(calculate_path, script_name)
            if os.path.exists(script_path):
                slurm_script_path = script_path
                break
        
        if slurm_script_path is None:
            return {
                "slurm_id": None,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "success": False,
                "error": "No SLURM script found (submit.sh, run.sh, or job.sh)",
                "status": "failed"
            }
        
        # Submit the SLURM job
        time.sleep(3)
        result = subprocess.run(['sbatch', slurm_script_path],
                              capture_output=True, text=True, cwd=calculate_path)

        if result.returncode == 0:
            # Extract the job ID from the sbatch output
            slurm_id = result.stdout.strip().split()[-1]
            
            return {
                "slurm_id": slurm_id,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "success": True,
                "error": None,
                "status": "submitted"
            }
        else:
            return {
                "slurm_id": None,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "success": False,
                "error": result.stderr,
                "status": "failed"
            }
            
    except Exception as e:
        return {
            "slurm_id": None,
            "calc_type": calc_type,
            "calculate_path": calculate_path,
            "success": False,
            "error": str(e),
            "status": "failed"
        }


def vasp_relaxation(calculation_id: str, work_dir: str, struct: Structure, 
                   kpoints: Kpoints, incar_dict: dict, attachment_path: Optional[str] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Submit a VASP structural relaxation calculation task

    Args:
        calculation_id: Calculation ID
        work_dir: Working directory
        struct: Crystal structure
        kpoints: K-point settings
        incar_dict: Additional INCAR parameters, merged with the defaults. Do not modify these on your own unless the user specifies it.
        attachment_path: Attachment path containing the SLURM script and other files

    Returns:
        Dict containing slurm_id, calc_type, calculate_path, success, error, status, etc.
    """
    Name = calculation_id
    calc_dir = os.path.abspath(f'{work_dir}/{Name}')
    if potcar_map is None:
        potcar_map = {}
    # Create the VASP input files
    # Manually derive the element list to ensure the order matches POSCAR
    poscar = Poscar(struct)
    unique_species = []
    for species in poscar.structure.species:
        species: Element
        if unique_species:
            if species.symbol != unique_species[-1]:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        else:
            if species.symbol not in potcar_map:
                potcar_map[species.symbol] = species.symbol
            unique_species.append(species.symbol)
    potcar_symbols = []
    for symbol in unique_species:
        potcar_symbols.append(potcar_map[symbol])

    vasp_input = VaspInput(
        poscar=poscar,
        incar=incar_dict,
        kpoints=kpoints,
        potcar=Potcar(potcar_symbols)
    )
    
    # Prepare the structural relaxation directory
    rlx_dir = os.path.join(calc_dir, "rlx/")
    os.makedirs(rlx_dir, exist_ok=True)
    vasp_input.write_input(rlx_dir)

    # Submit the SLURM job
    return _submit_slurm_job("relaxation", rlx_dir, attachment_path)


def vasp_scf(calculation_id: str, work_dir: str, struct: Structure, 
            kpoints: Kpoints, incar_dict: dict, chgcar_path: Optional[str] = None, 
            wavecar_path: Optional[str] = None, attachment_path: Optional[str] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Submit a VASP self-consistent field (SCF) calculation task

    Args:
        calculation_id: Calculation ID
        work_dir: Working directory
        struct: Crystal structure
        kpoints: K-point settings
        incar_dict: Additional INCAR parameters, merged with the defaults. Do not modify these on your own unless the user specifies it.
        chgcar_path: Path to the CHGCAR file
        wavecar_path: Path to the WAVECAR file
        attachment_path: Attachment path containing the SLURM script and other files

    Returns:
        Dict containing slurm_id, calc_type, calculate_path, success, error, status, etc.
    """
    Name = calculation_id
    calc_dir = os.path.abspath(f'{work_dir}/{Name}')
    if potcar_map is None:
        potcar_map = {}
    # Create the VASP input files
    # Manually derive the element list to ensure the order matches POSCAR
    poscar = Poscar(struct)
    unique_species = []
    for species in poscar.structure.species:
        species: Element
        if unique_species:
            if species.symbol != unique_species[-1]:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        else:
            if species.symbol not in potcar_map:
                potcar_map[species.symbol] = species.symbol
            unique_species.append(species.symbol)
    potcar_symbols = []
    for symbol in unique_species:
        potcar_symbols.append(potcar_map[symbol])

    vasp_input = VaspInput(
        poscar=poscar,
        incar=incar_dict,
        kpoints=kpoints,
        potcar=Potcar(potcar_symbols)
    )

    # Prepare the SCF calculation directory
    scf_dir = os.path.join(calc_dir, "scf/")
    os.makedirs(scf_dir, exist_ok=True)
    vasp_input.write_input(scf_dir)

    # Copy over related files
    if chgcar_path is not None and os.path.exists(chgcar_path):
        shutil.copy2(chgcar_path, os.path.join(scf_dir, "CHGCAR"))
    if wavecar_path is not None and os.path.exists(wavecar_path):
        shutil.copy2(wavecar_path, os.path.join(scf_dir, "WAVECAR"))

    # Submit the SLURM job
    return _submit_slurm_job("scf", scf_dir, attachment_path)


def vasp_nscf(calculation_id: str, work_dir: str, struct: Structure, 
             kpoints: Kpoints, incar_dict: dict, chgcar_path: str, 
             wavecar_path: Optional[str] = None, attachment_path: Optional[str] = None, 
             potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Submit a VASP non-self-consistent field (NSCF) calculation task (band structure calculation)

    Args:
        calculation_id: Calculation ID
        work_dir: Working directory
        struct: Crystal structure
        kpoints: K-point settings
        incar_dict: Additional INCAR parameters, merged with the defaults. Do not modify these on your own unless the user specifies it.
        chgcar_path: Path to the CHGCAR file
        wavecar_path: Path to the WAVECAR file
        attachment_path: Attachment path containing the SLURM script and other files
        potcar_map: POTCAR mapping dictionary

    Returns:
        Dict containing slurm_id, calc_type, calculate_path, success, error, status, etc.
    """
    Name = calculation_id
    calc_dir = os.path.abspath(f'{work_dir}/{Name}')
    if potcar_map is None:
        potcar_map = {}
    # Create the VASP input files
    # Manually derive the element list to ensure the order matches POSCAR
    poscar = Poscar(struct)
    unique_species = []
    for species in poscar.structure.species:
        species: Element
        if unique_species:
            if species.symbol != unique_species[-1]:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        else:
            if species.symbol not in potcar_map:
                potcar_map[species.symbol] = species.symbol
            unique_species.append(species.symbol)
    potcar_symbols = []
    for symbol in unique_species:
        potcar_symbols.append(potcar_map[symbol])

    vasp_input = VaspInput(
        poscar=poscar,
        incar=incar_dict,
        kpoints=kpoints,
        potcar=Potcar(potcar_symbols)
    )
    
    # Prepare the band structure calculation directory
    band_dir = os.path.join(calc_dir, "band/")
    os.makedirs(band_dir, exist_ok=True)
    vasp_input.write_input(band_dir)

    # Copy over related files
    if os.path.exists(chgcar_path):
        shutil.copy2(chgcar_path, os.path.join(band_dir, "CHGCAR"))
    if wavecar_path is not None and os.path.exists(wavecar_path):
        shutil.copy2(wavecar_path, os.path.join(band_dir, "WAVECAR"))

    # Submit the SLURM job
    return _submit_slurm_job("nscf", band_dir, attachment_path)


def check_status(calc_dict: dict[str, dict[str, Any]]) -> Dict[str, Any]:
    """
    Check SLURM job status and return the calculation results

    Args:
        calc_dict: {
            calc_id: {
                "slurm_id": slurm_id,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "status": status
            }
        }

    Returns:
        Dict containing the status and result of each job
    """

    for calc_id, job_info in calc_dict.items():
        slurm_id = job_info["slurm_id"]
        calc_type = job_info["calc_type"]
        calculate_path = job_info["calculate_path"]

        try:
            # Check the SLURM job status
            time.sleep(3)

            result = subprocess.run(
                ["squeue", "-j", slurm_id, "--noheader"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                # Job is still running
                job_status = "running"
                job_result = {}

            else:
                # Job has left squeue; use scontrol to check its final state
                time.sleep(3)

                scontrol_result = subprocess.run(
                    ["scontrol", "show", "job", slurm_id, "-o"],
                    capture_output=True,
                    text=True
                )

                if (
                    scontrol_result.returncode == 0
                    and scontrol_result.stdout.strip()
                ):
                    state = ""

                    # Example:
                    # JobId=6 ... JobState=COMPLETED ...
                    for field in scontrol_result.stdout.strip().split():
                        if field.startswith("JobState="):
                            state = field.split("=", 1)[1]
                            break

                    if state == "COMPLETED":
                        job_status = "completed"

                        job_result = _read_calculation_result(
                            calc_type,
                            calculate_path
                        )

                    elif state == "TIMEOUT":
                        job_status = "timeout"
                        job_result = {
                            "error": "SLURM job timed out"
                        }

                    elif state in {
                        "FAILED",
                        "CANCELLED",
                        "NODE_FAIL",
                        "OUT_OF_MEMORY",
                        "BOOT_FAIL",
                        "DEADLINE",
                        "PREEMPTED",
                    }:
                        err_str = """ -----------------------------------------------------------------------------
|                                                                             |
|     EEEEEEE  RRRRRR   RRRRRR   OOOOOOO  RRRRRR      ###     ###     ###     |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###     |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###     |
|     EEEEE    RRRRRR   RRRRRR   O     O  RRRRRR       #       #       #      |
|     E        R   R    R   R    O     O  R   R                               |
|     E        R    R   R    R   O     O  R    R      ###     ###     ###     |
|     EEEEEEE  R     R  R     R  OOOOOOO  R     R     ###     ###     ###     |"""

                        try:
                            log_path = os.path.join(
                                calculate_path,
                                "log"
                            )

                            if not os.path.exists(log_path):
                                log_path = os.path.join(
                                    calculate_path,
                                    "OUTCAR"
                                )

                            with open(log_path, "r") as f:
                                content = f.read()

                            if err_str in content:
                                log_content = content.split(
                                    err_str,
                                    1
                                )[1]
                            else:
                                log_content = content

                        except Exception:
                            log_content = (
                                "SLURM job failed without any error message"
                            )

                        job_status = "failed"
                        job_result = {
                            "error": log_content
                        }

                    elif state:
                        job_status = state.lower()
                        job_result = {
                            "error": (
                                f"SLURM job exited with state: {state}"
                            )
                        }

                    else:
                        job_status = "unknown"
                        job_result = {
                            "error": (
                                "Cannot find JobState from scontrol"
                            )
                        }

                else:
                    job_status = "unknown"
                    job_result = {
                        "error": (
                            "Cannot determine job status: "
                            f"{scontrol_result.stderr.strip()}"
                        )
                    }

            calc_dict[calc_id].update(job_result)
            calc_dict[calc_id]["status"] = job_status

        except Exception as e:
            calc_dict[calc_id] = {
                "slurm_id": slurm_id,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "status": "error",
                "error": str(e)
            }

    return calc_dict

def _read_calculation_result(calc_type: str, calculate_path: str) -> Dict[str, Any]:
    """
    Read the calculation result based on the calculation type
    """
    try:
        if calc_type == "relaxation":
            # Read the structural relaxation result
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            contcar = Poscar.from_file(os.path.join(calculate_path, "CONTCAR"))
            
            return {
                "structure": contcar.structure,
                "total_energy": vasprun.final_energy,
                "max_force": np.max(np.linalg.norm(vasprun.ionic_steps[-1]['forces'], axis=1)),
                "stress": vasprun.ionic_steps[-1]['stress'],
                "ionic_steps": len(vasprun.ionic_steps),
                "status": "completed"
            }
            
        elif calc_type == "scf":
            # Read the SCF calculation result
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            
            return {
                "structure": vasprun.final_structure,
                "total_energy": vasprun.final_energy,
                "efermi": vasprun.efermi,
                "band_gap": vasprun.get_band_structure().get_band_gap(),
                "dos": vasprun.complete_dos,
                "eigen_values": vasprun.eigenvalues,
                "is_metal": vasprun.get_band_structure().is_metal(),
                "status": "completed"
            }
            
        elif calc_type == "nscf":
            # Read the band structure calculation result
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            bs = vasprun.get_band_structure()
            
            return {
                "structure": vasprun.final_structure,
                "band_structure": bs,
                "efermi": vasprun.efermi,
                "dos": vasprun.complete_dos,
                "eigenvalues": vasprun.eigenvalues,
                "is_metal": bs.is_metal(),
                "band_gap": bs.get_band_gap(),
                "cbm": bs.get_cbm(),
                "vbm": bs.get_vbm(),
                "success": True
            }
        else:
            return {
                "success": False,
                "error": f"Unknown calculation type: {calc_type}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def cancel_slurm_job(slurm_id: str) -> Dict[str, Any]:
    """
    Cancel a SLURM job
    """
    try:
        subprocess.run(['scancel', slurm_id], capture_output=True, text=True)
        return {"success": True, "message": f"SLURM job {slurm_id} cancelled"}
    except Exception as e:
        return {"success": False, "error": str(e)}