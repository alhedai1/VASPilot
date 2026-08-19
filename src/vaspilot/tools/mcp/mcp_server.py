import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from .python_plot import safe_execute_plot_code
import uuid
from fastmcp import FastMCP
from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints
from ase.dft.kpoints import BandPath
import yaml
import math
import numpy as np
from .vasp_calculate import vasp_relaxation, vasp_scf, vasp_nscf, check_status, cancel_slurm_job
from .struct_tools import search_materials_project, analyze_crystal_structure, create_crystal_structure, make_supercell, rotate_structure, symmetrize_structure, scale_structure
from .sqlite_database import VaspCalculationDB
def main(config_path: str = None, port: int = 8933, host: str = "0.0.0.0"):

    # Load the config file
    if config_path is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent.parent
        config_path = f"{project_root}/configs/mcp_config.yaml"

    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)

    db_path = settings['db_path']
    attachment_path = settings['attachment_path']
    mp_api_key = settings['mp_api_key']
    structure_path = settings['structure_path']

    # Initialize the SQLite database
    db = VaspCalculationDB(db_path=db_path)

    mcp = FastMCP("VASP Agent")

    def write_record(calculation_id: str, data: dict):
        """Write a calculation record to the SQLite database"""
        db.write_record(calculation_id, data)
        return

    def read_record(calculation_id: str):
        """Read a calculation record from the SQLite database"""
        return db.read_record(calculation_id)

    def extract_calculation_data(calculation_ids: List[str]) -> Dict[str, Any]:
        """Extract data from calculation records"""
        data = {}
        for calc_id in calculation_ids:
            record = read_record(calc_id)
            if record is not None:
                data[calc_id] = record
            else:
                data[calc_id] = None
        return data

    def _to_json_safe(value):
        """Recursively convert numpy scalars/arrays to native Python types so MCP tool
        output (which is validated against a JSON outputSchema) can be serialized."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): _to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_json_safe(v) for v in value]
        return value

    def extract_llm_friendly_result(data: Dict[str, Any]) -> Dict[str, Any]:
        llm_friendly_result = {
            "status": data.get("status", "unknown"),
            "error": data.get("error"),
            "slurm_id": data.get("slurm_id"),
            "calculate_path": data.get("calculate_path"),
            "calc_type": data.get("calc_type"),
        }

        if data.get("calc_type") == "relaxation":
            llm_friendly_result.update({
                "total_energy": data.get("total_energy"),
                "max_force": data.get("max_force"),
                "stress": data.get("stress"),
                "ionic_steps": data.get("ionic_steps")
            })

        elif data.get("calc_type") == "scf":
            llm_friendly_result.update({
                "total_energy": data.get("total_energy"),
                "efermi": data.get("efermi"),
                "band_gap": data.get("band_gap"),
                "is_metal": data.get("is_metal")
            })

        elif data.get("calc_type") == "nscf":
            llm_friendly_result.update({
                "efermi": data.get("efermi"),
                "band_gap": data.get("band_gap"),
                "is_metal": data.get("is_metal")
            })

        return _to_json_safe(llm_friendly_result)

    @mcp.tool(name="vasp_relaxation")
    async def vasp_relaxation_tool(structure_path: str, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Submit a VASP structural relaxation job.
        
        Args:
            structure_path: Path to the structure file (supports CIF, POSCAR, etc.).
            incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
            kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
            potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_d", "Se": "Se"}. Use None unless explicitly specified by the user.
        Returns:
            A dict containing the submission result with keys:
            - calculation_id: Unique calculation identifier
            - slurm_id: SLURM job ID
            - success: Whether submission succeeded
            - error: Error message, if any
            - status: Job status ("pending"/"failed")
        """
        # Convert input parameters

        # Generate a random UUID
        calculation_id = str(uuid.uuid4())
        struct = Structure.from_file(structure_path)
        if kpoint_num is None:
            factor = 40 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
            kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
            kpoint_num = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
        kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
        incar = {}
        incar.update(settings['VASP_default_INCAR']['relaxation'])
        if incar_tags is not None:
            incar.update(incar_tags)

        # Run the calculation
        result = vasp_relaxation(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            attachment_path=attachment_path,
            potcar_map=potcar_map
        )

        # Save the record
        result['calculation_id'] = calculation_id
        write_record(calculation_id, result)
        
        llm_friendly_result = {
            'calculation_id': calculation_id,
            'slurm_id': result['slurm_id'],
            'success': result['success'],
            'error': result['error'],
            'status': result['status'],
            'calculate_path': result['calculate_path']
        }
        return llm_friendly_result

    @mcp.tool(name="vasp_scf")
    async def vasp_scf_tool(restart_id: Optional[str] = None, structure_path: Optional[str] = None, soc: bool=True, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Submit a VASP self-consistent field (SCF) job.
        
        Args:
            restart_id: ID of a previous calculation. If provided, reuse its structure and charge density.
            structure_path: Path to the structure file; required when restart_id is not provided.
            soc: Whether to include spin–orbit coupling. Defaults to True.
            incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
            kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
            potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
        Returns:
            A dict containing the submission result with keys:
            - calculation_id: Unique calculation identifier
            - slurm_id: SLURM job ID
            - success: Whether submission succeeded
            - error: Error message, if any
            - status: Job status ("pending"/"failed")
        """
        # Convert input parameters

        # Generate a random UUID
        calculation_id = str(uuid.uuid4())
        if restart_id is not None:
            restart_record = read_record(restart_id)
            if restart_record is None:
                return {"success": False, "error": f"Restart record {restart_id} not found"}
            struct = restart_record['structure']
            chgcar_path = os.path.join(restart_record['calculate_path'], "CHGCAR")
            wavecar_path = os.path.join(restart_record['calculate_path'], "WAVECAR")
        else:
            if structure_path is None:
                return {"success": False, "error": "structure_path is required when restart_id is not provided"}
            else:
                try:
                    struct = Structure.from_file(structure_path)
                except Exception as e:
                    return {"success": False, "error": f"Failed to read structure from {structure_path}: {e}"}
            chgcar_path = None
            wavecar_path = None
        if kpoint_num is None:
            factor = 40 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
            kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
            kpoint_num = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
        kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
        incar = {}
        if soc:
            incar.update(settings['VASP_default_INCAR']['scf_soc'])
        else:
            incar.update(settings['VASP_default_INCAR']['scf_nsoc'])
        if incar_tags is not None:
            incar.update(incar_tags)
        
        # Run the calculation
        result = vasp_scf(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            chgcar_path=chgcar_path,
            wavecar_path=wavecar_path,
            attachment_path=attachment_path,
            potcar_map=potcar_map
        )

        # Save the record
        result['calculation_id'] = calculation_id
        result['soc'] = soc
        result['incar_tags'] = incar_tags
        result['restart_id'] = restart_id
        write_record(calculation_id, result)
        
        llm_friendly_result = {
            'calculation_id': calculation_id,
            'slurm_id': result['slurm_id'],
            'success': result['success'],
            'error': result['error'],
            'status': result['status'],
            'calculate_path': result['calculate_path']
        }
        return llm_friendly_result

    @mcp.tool(name="vasp_nscf_kpath")
    async def vasp_nscf_kpath_tool(restart_id: str, soc: bool=True, incar_tags: Optional[Dict] = None, kpath: Optional[str] = None, n_kpoints: Optional[int] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Submit a VASP non-self-consistent field (NSCF) job for band structure. INCAR parameters should be consistent with the preceding SCF job where possible.
        
        Args:
            restart_id: ID of the preceding SCF calculation (required) to obtain converged charge density and wavefunction.
            soc: Whether to include spin–orbit coupling. Defaults to True.
            incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
            kpath: K-point path. Options:
                - None: Use the auto-generated high-symmetry path from pymatgen.
                - str: User-specified path, e.g., "GMKG".
                Use None unless explicitly specified by the user.
            n_kpoints: Number of points per segment along the k-path.
            potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
        Returns:
            A dict containing the submission result with keys:
            - calculation_id: Unique calculation identifier
            - slurm_id: SLURM job ID
            - success: Whether submission succeeded
            - error: Error message, if any
            - status: Job status ("pending"/"failed")
        """
        # Generate a random UUID
        calculation_id = str(uuid.uuid4())
        
        # Get the structure and preceding calculation files
        scf_record = read_record(restart_id)
        if scf_record is None:
            return {"success": False, "error": f"SCF record {restart_id} not found"}
        struct: Structure = scf_record['structure']
        chgcar_path = os.path.join(scf_record['calculate_path'], "CHGCAR")
        wavecar_path = os.path.join(scf_record['calculate_path'], "WAVECAR")
        
        # Set the k-point path
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        kpath_obj = HighSymmKpath(struct)
        if kpath_obj.kpath is None:
            return {"success": False, "error": "Failed to generate k-path for the structure"}

        n_kpoints = 16 if n_kpoints is None else n_kpoints
        if kpath is None:
            # Use the auto-generated high-symmetry path from pymatgen
            kpts = Kpoints.automatic_linemode(n_kpoints, kpath_obj)
        else:
            # Use the user-specified path
            kpts_ase: BandPath = struct.to_ase_atoms().get_cell().bandpath(kpath, npoints=n_kpoints, eps=1e-2)
            high_sym_points = []
            labels = []
            high_sym_points.append(kpts_ase.special_points[kpath[0]])
            labels.append(kpath[0])
            kpath_list = list(kpath)
            for key in kpath_list[1:-1]:
                high_sym_points.append(kpts_ase.special_points[key])
                labels.append(key)
                high_sym_points.append(kpts_ase.special_points[key])
                labels.append(key)
            high_sym_points.append(kpts_ase.special_points[kpath[-1]])
            labels.append(kpath[-1])
            kpts = Kpoints(
                comment="User specified k-path",
                style=Kpoints.supported_modes.Line_mode,
                num_kpts=n_kpoints,
                kpts=high_sym_points,
                labels=labels,
                coord_type="Reciprocal"
            )
        
        # Set INCAR
        incar = {}
        if soc:
            incar.update(settings['VASP_default_INCAR']['nscf_soc'])
        else:
            incar.update(settings['VASP_default_INCAR']['nscf_nsoc'])
        if incar_tags is not None:
            incar.update(incar_tags)
        
        # Run the calculation
        result = vasp_nscf(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            chgcar_path=chgcar_path,
            wavecar_path=wavecar_path,
            attachment_path=attachment_path,
            potcar_map=potcar_map
        )

        # Save the record
        result['calculation_id'] = calculation_id
        result['soc'] = soc
        result['incar_tags'] = incar_tags
        result['restart_id'] = restart_id
        result['kpath'] = kpath
        result['n_kpoints'] = n_kpoints
        write_record(calculation_id, result)
        
        llm_friendly_result = {
            'calculation_id': calculation_id,
            'slurm_id': result['slurm_id'],
            'success': result['success'],
            'error': result['error'],
            'status': result['status'],
            'calculate_path': result['calculate_path']
        }
        return llm_friendly_result

    @mcp.tool(name="vasp_nscf_uniform")
    async def vasp_nscf_uniform_tool(restart_id: str, soc: bool=True, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Submit a VASP non-self-consistent field (NSCF) job for band structure. INCAR parameters should be consistent with the preceding SCF job where possible.
        
        Args:
            restart_id: ID of the preceding SCF calculation (required) to obtain converged charge density and wavefunction.
            soc: Whether to include spin–orbit coupling. Defaults to True.
            incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
            kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
            potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
        Returns:
            A dict containing the submission result with keys:
            - calculation_id: Unique calculation identifier
            - slurm_id: SLURM job ID
            - success: Whether submission succeeded
            - error: Error message, if any
            - status: Job status ("pending"/"failed")
        """
        # Generate a random UUID
        calculation_id = str(uuid.uuid4())
        
        # Get the structure and preceding calculation files
        scf_record = read_record(restart_id)
        if scf_record is None:
            return {"success": False, "error": f"SCF record {restart_id} not found"}
        struct: Structure = scf_record['structure']
        chgcar_path = os.path.join(scf_record['calculate_path'], "CHGCAR")
        wavecar_path = os.path.join(scf_record['calculate_path'], "WAVECAR")
        
        # Set k-points

        if kpoint_num is None:
            factor = 100 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
            kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
            kpoint_num = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
        kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
        
        # Set INCAR
        incar = {}
        if soc:
            incar.update(settings['VASP_default_INCAR']['nscf_soc'])
        else:
            incar.update(settings['VASP_default_INCAR']['nscf_nsoc'])
        if incar_tags is not None:
            incar.update(incar_tags)
        
        # Run the calculation
        result = vasp_nscf(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            chgcar_path=chgcar_path,
            wavecar_path=wavecar_path,
            attachment_path=attachment_path,
            potcar_map=potcar_map
        )

        # Save the record
        result['calculation_id'] = calculation_id
        result['soc'] = soc
        result['incar_tags'] = incar_tags
        result['restart_id'] = restart_id
        result['kpoint_num'] = kpoint_num
        write_record(calculation_id, result)
        
        llm_friendly_result = {
            'calculation_id': calculation_id,
            'slurm_id': result['slurm_id'],
            'success': result['success'],
            'error': result['error'],
            'status': result['status'],
            'calculate_path': result['calculate_path']
        }
        return llm_friendly_result

    @mcp.tool(name="check_calculation_status")
    async def check_calculation_status_tool(calculation_ids: List[str]) -> Dict[str, Any]:
        """
        Check the status of calculation jobs and return results.
        
        Args:
            calculation_ids: List of calculation IDs to check
        
        Returns:
            A dict mapping each calculation_id to the following structure:
            {
                calculation_id: {
                    "slurm_id": "12345",
                    "calc_type": "relaxation",
                    "calculate_path": "/path/to/calculation",
                    "status": "running/completed/failed/error",
                    ... other result fields
                }
            }
        """
        # Gather calculation info from records
        calc_dict = {}
        llm_friendly_result = {}
        
        # Collect valid calculation records
        for calc_id in calculation_ids:
            record = read_record(calc_id)
            if record is not None:
                calc_dict[calc_id] = record
        
        # Check SLURM status for calculations that have records
        if calc_dict:
            updated_results = check_status(calc_dict)
            # Update the record
            for calc_id, result in updated_results.items():
                write_record(calc_id, result)
                calc_dict[calc_id] = result
        
        # Build the return result for all calculation IDs
        for calc_id in calculation_ids:
            if calc_id in calc_dict:
                data = calc_dict[calc_id]
                llm_friendly_result[calc_id] = extract_llm_friendly_result(data)
            else:
                llm_friendly_result[calc_id] = {
                    "error": f"Calculation record {calc_id} not found",
                    "status": "unknown"
                }
        
        return llm_friendly_result

    @mcp.tool(name="python_plot")
    async def python_plot_tool(calculation_ids: List[str], plot_code: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute Python plotting code based on the specified calculation result data.
        Note: Do not use plt.show() or plt.savefig() in plot_code, and do NOT call plt.close().
        For multiple figures, call this tool multiple times, otherwise images will be overwritten.

        Args:
            calculation_ids: List of calculation IDs to extract result data
            plot_code: Python plotting code. The data will be provided as a dict named `data`,
                       where keys are calculation_id and values are the corresponding results.
                    
                       Available libraries and objects:
                       - matplotlib.pyplot (as plt)
                       - numpy (as np)
                       - pandas (as pd)
                       - pymatgen.core.Structure (as Structure)
                       - pymatgen.core.Element (as Element)
                       - pymatgen.core.Lattice (as Lattice)
                       - pymatgen.electronic_structure.bandstructure.BandStructure (as BandStructure)
                       - pymatgen.electronic_structure.dos.CompleteDos (as CompleteDos)
                       - pymatgen.io.vasp.Vasprun (as Vasprun)
                    
                       Data format notes:
                       - `data` is a dict mapping calculation_id to its result dict
                       - Results contain raw pymatgen objects; you can directly use pymatgen methods for processing and plotting
                       - Common fields include:
                           * 'structure': pymatgen.core.Structure
                           * 'band_structure': pymatgen.electronic_structure.bandstructure.BandStructure  
                           * 'dos': pymatgen.electronic_structure.dos.CompleteDos
                           * 'total_energy': float
                           * 'efermi': float
                           * 'band_gap': dict with keys "energy", "direct", "transition"
                           * 'stress': float
                           * 'eigen_values': numpy.ndarray
                    
                       Example 1: Plot band structure (use this when there is no special requirement)
                       ```python
                       # Get the first calculation data
                       band_calc_id = list(data.keys())[0]
                       calc_data = data[band_calc_id]
                       
                       # Plot band structure
                       if 'band_structure' in calc_data:
                           bs = calc_data['band_structure']  # BandStructure object
                           plt.figure(figsize=(10, 6))
                           
                           # Use pymatgen built-in plotter
                           from pymatgen.electronic_structure.plotter import BSPlotter
                           plotter = BSPlotter(bs)
                           plotter.get_plot(ylim=[-2, 2])
                           plt.axhline(y=0, color='r', linestyle='--', label='Fermi level')
                           plt.ylabel('Energy - E_fermi (eV)')
                           plt.xlabel('k-point')
                           plt.legend()
                       ```
                       Example 2: Plot band structure (for special requirements, e.g., overlay two band structures)
                       ```python
                       # Get the first calculation data
                       band_calc_id = list(data.keys())[0]
                       calc_data = data[band_calc_id]
                       
                       # Plot band structure
                       if 'band_structure' in calc_data:
                           bs = calc_data['band_structure']  # BandStructure object
                           # Extract plot data via pymatgen
                           from pymatgen.electronic_structure.plotter import BSPlotter
                           plotter = BSPlotter(bs)
                           ax = plt.gca()
                           data = plotter.bs_plot_data(bs)
                           # Draw
                           for spin in bs.bands:
                               ls = "-" if str(spin) == "1" else "--"
                               for dist, ene in zip(data["distances"], data["energy"][str(spin)], strict=True):
                                   ax.plot(dist, ene.T, ls=ls, c="blue")

                           ax.set_xticks(data["ticks"]["distance"], data["ticks"]["label"])
                           for distance in data["ticks"]["distance"]:
                               ax.axvline(x=distance, color='black', linewidth=0.5)
                           plt.axhline(y=0, color='r', linestyle='--', label='Fermi level')
                           plt.xlim(np.min(data["distances"]), np.max(data["distances"]))
                           plt.ylim(-5, 5)
                           plt.ylabel('Energy - E_fermi (eV)')
                           plt.xlabel('k points')
                       ```
                       Example 3: Plot DOS
                       ```python
                       # Get DOS calculation data
                       dos_calc_id = list(data.keys())[0]
                       dos_data = data[dos_calc_id]
                       # Plot DOS
                       if 'dos' in dos_data:
                           dos = dos_data['dos']  # CompleteDos object
                           plt.figure(figsize=(8, 6))
                           # Use pymatgen built-in plotter
                           from pymatgen.electronic_structure.plotter import DosPlotter
                           plotter = DosPlotter()
                           plotter.add_dos("Total DOS", dos)
                           plotter.get_plot(xlim=[-5, 5])
                       ```
            description: Optional description for the figure
        
        Returns:
            A dict with keys:
            - success: Whether execution succeeded
            - error: Error message, if any
            - plot_path: Path to the generated image file
            - calculation_data_summary: Summary of the data used
        """
        
        # Extract calculation data
        calculation_data = extract_calculation_data(calculation_ids)
        # Check whether there is any valid calculation data
        valid_data = {k: v for k, v in calculation_data.items() if v is not None}
        if not valid_data:
            return {
                'success': False,
                'error': f'No valid calculation data found. Provided calculation IDs: {calculation_ids}',
                'plot_path': None,
                'image_base64': None,
                'calculation_data_summary': None
            }
        
        # Create a data summary
        data_summary = {}
        for calc_id, data in valid_data.items():
            if data:
                summary = {
                    'calculation_id': calc_id,
                    'success': data.get('success', False),
                    'total_energy': data.get('total_energy', None),
                    'available_keys': list(data.keys())
                }
                data_summary[calc_id] = summary
        
        # Execute the plotting code
        success, result, image_base64 = safe_execute_plot_code(plot_code, valid_data, settings['work_dir'])
        if success:
            return {
                'success': True,
                'error': None,
                'plot_path': result,
                'description': description
            }
        else:
            return {
                'success': False,
                'error': result,
                'plot_path': None,
                'calculation_data_summary': data_summary,
                'description': description
            }

    @mcp.tool(name="search_materials_project")
    async def search_materials_project_tool(search_criteria: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """
        Args:
            search_criteria: Search filters with the following keys:
                - material_id: str, e.g., "mp-1234"
                - formula: str, e.g., "TiO2"
                - elements: List[str], e.g., ["Ti", "O"]
                - exclude_elements: List[str], elements to exclude
                - band_gap: Tuple[float, float], band gap range (min, max), e.g., (1.0, 3.0)
                - energy_above_hull: Tuple[float, float], formation energy range (min, max)
                - num_sites: Tuple[int, int], number of sites range (min, max)
                - spacegroup_number: int, space group number
                - crystal_system: str, one of "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"
                - is_gap_direct: bool, whether the band gap is direct
            limit: Maximum number of results to return
            
        Returns:
            Dict with search results and download status
        """
        result = search_materials_project(api_key=mp_api_key, search_criteria=search_criteria, download_path=structure_path, limit=limit)
        return result

    @mcp.tool(name="analyze_crystal_structure")
    async def analyze_crystal_structure_tool(struct_path: str) -> Dict[str, Any]:
        """
        Args:
            struct_path: Structure input; can be a file path or a pymatgen Structure object
        """
        return analyze_crystal_structure(struct_path)

    @mcp.tool(name="create_crystal_structure")
    async def create_crystal_structure_tool(positions: List[List[float]], elements: List[str], lattice_vectors: List[List[float]], cartesian: bool) -> Dict[str, Any]:
        """
        Create a crystal structure given atomic positions, elements, and lattice vectors, and save it to a file.
        Args:
            positions: Atomic positions, e.g., [[x1, y1, z1], [x2, y2, z2], ...]
            elements: List of element symbols, e.g., ["Li", "F"]
            lattice_vectors: Lattice vectors, e.g., [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]]
            cartesian: Whether positions are in Cartesian coordinates
        Returns:
            Dict containing the path to the created structure file
        """
        return create_crystal_structure(np.array(positions), elements, np.array(lattice_vectors), cartesian, structure_path)

    @mcp.tool(name="make_supercell")
    async def make_supercell_tool(struct_path: str, supercell_matrix: List[List[int]]) -> Dict[str, Any]:
        """
        Args:
            struct_path: Structure input; can be a file path or a pymatgen Structure object
            supercell_matrix: 3 by 3 matrix, supercell matrix.
        return:
            Dict containing the path to the created structure file
        """
        return make_supercell(struct_path, supercell_matrix)

    @mcp.tool(name="scale_structure")
    async def scale_structure_tool(struct_path: str, scale_factors: List[int]) -> Dict[str, Any]:
        """
        Args:
            struct_path: Structure input; can be a file path or a pymatgen Structure object
            scale_factors: Scale factors, e.g., [2, 2, 1]
        return:
            Dict containing the path to the created structure file
        """
        return scale_structure(struct_path, scale_factors)

    @mcp.tool(name="symmetrize_structure")
    async def symmetrize_structure_tool(struct_path: str) -> Dict[str, Any]:
        """
        Args:
            struct_path: Structure input; can be a file path or a pymatgen Structure object
        """
        return symmetrize_structure(struct_path)

    @mcp.tool(name="list_calculations")
    async def list_calculations_tool(calc_type: Optional[str] = None, status: Optional[str] = None, limit: Optional[int] = 50) -> Dict[str, Any]:
        """
        List calculation records.
        
        Args:
            calc_type: Filter by calculation type ("relaxation", "scf", "nscf")
            status: Filter by status ("pending", "running", "completed", "failed", "error")
            limit: Maximum number of records to return, default 50
        
        Returns:
            result_dict: A dict containing the list of records
        """
        try:
            calculations = db.list_calculations(calc_type=calc_type, status=status, limit=limit)
            return {
                "success": True,
                "calculations": calculations,
                "total_count": len(calculations)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "calculations": []
            }

    @mcp.tool(name="get_database_statistics")
    async def get_database_statistics_tool() -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Database statistics including total records, distribution by type and status, etc.
        """
        try:
            stats = db.get_statistics()
            return {
                "success": True,
                "statistics": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }

    @mcp.tool(name="delete_calculation")
    async def delete_calculation_tool(calculation_id: str) -> Dict[str, Any]:
        """
        Delete a calculation record.
        
        Args:
            calculation_id: ID of the record to delete
        
        Returns:
            Result of the deletion operation
        """
        try:
            success = db.delete_record(calculation_id)
            if success:
                return {
                    "success": True,
                    "message": f"Successfully deleted calculation record {calculation_id}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Calculation record {calculation_id} does not exist"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error while deleting calculation record: {str(e)}"
            }

    @mcp.tool(name="check_files_exist")
    async def check_files_exist(file_paths: list[str]) -> Dict[str, Any]:
        """
        Check whether each file in the list exists.
        
        Args:
            file_paths: List of file paths to check
        
        Returns:
            exist_dict: A dict of path -> bool, e.g.,
            {
                "file_path1": True,
                "file_path2": True,
                ......
            }
        """

        files_exist = {}
        for path in file_paths:
            files_exist[path] = os.path.exists(path)
        return files_exist

    @mcp.tool(name="read_calc_results_from_db")
    async def read_calc_results_from_db(calc_ids: list[str]) -> Dict[str, Any]:
        """
        Read calculation records.
        Args:
            calc_ids: List of calculation IDs to read
        
        Returns:
            A dict mapping each calculation_id to the following structure:
            {
                calculation_id: {
                    "slurm_id": "12345",
                    "calc_type": "relaxation",
                    "calculate_path": "/path/to/calculation",
                    "status": "running/completed/failed/error",
                    ... other result fields
                }
            }
        """
        llm_friendly_results = {}
        for calc_id in calc_ids:
            data = read_record(calc_id)
            if data is not None:
                llm_friendly_results[calc_id] = extract_llm_friendly_result(data)
            else:
                llm_friendly_results[calc_id] = {"error": "calc_id not found!"}
        return llm_friendly_results

    @mcp.tool(name="cancel_slurm_job")
    async def cancel_slurm_job_tool(calc_ids: list[str]) -> Dict[str, Any]:
        """
        Cancel SLURM jobs.
        
        Args:
            calc_ids: List of calculation IDs to cancel
        
        Returns:
            Result of the cancellation operations
        """
        result_dict = {}
        for calc_id in calc_ids:
            try:
                data = read_record(calc_id)
                if data is not None:
                    if data.get("status") == "running":
                        result = cancel_slurm_job(data.get("slurm_id"))
                        data["status"] = "cancelled"
                        write_record(calc_id, data)
                        result_dict[calc_id] = result
                    else:
                        result_dict[calc_id] = {"success": True, "message": f"SLURM job {data.get('slurm_id')} is not running"}
            except Exception as e:
                result_dict[calc_id] = {"success": False, "error": str(e)}
            
        return result_dict

    mcp.run(transport="streamable-http", host=host, port=port, stateless_http=True)

if __name__ == '__main__':
    main()
