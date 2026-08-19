import os
import traceback
from typing import Dict, Any, List, Optional, Union
import numpy as np
from pymatgen.core import Structure, Lattice
from mp_api.client import MPRester
from pymatgen.transformations.advanced_transformations import SupercellTransformation
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import uuid
from pymatgen.io.vasp import Poscar

def analyze_crystal_structure(struct_input: Union[str, Structure]) -> Dict[str, Any]:
    """
    Analyze the space group and chemical formula of a crystal structure

    Args:
        struct_input: Structure input, either a file path or a pymatgen Structure object

    Returns:
        Dict containing space group info, chemical formula, lattice parameters, etc.
    """

    try:
        # Process the input argument
        if isinstance(struct_input, str):
            # If it is a file path
            if os.path.exists(struct_input):
                struct = Structure.from_file(struct_input)
            else:
                return {
                    "success": False,
                    "error": f"File does not exist: {struct_input}",
                    "space_group": None,
                    "chemical_formula": None,
                    "lattice_parameters": None
                }
        elif isinstance(struct_input, Structure):
            struct = struct_input
        else:
            return {
                "success": False,
                "error": "Unsupported input type; please provide a file path or a pymatgen Structure object",
                "space_group": None,
                "chemical_formula": None,
                "lattice_parameters": None
            }

        # Use pymatgen to analyze the space group
        spg_analyzer = SpacegroupAnalyzer(struct)
        space_group = spg_analyzer.get_space_group_symbol()
        space_group_number = spg_analyzer.get_space_group_number()

        # Get the chemical formula
        chemical_formula = struct.composition.reduced_formula

        # Get the lattice parameters
        lattice = struct.lattice
        lattice_parameters = {
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
            "volume": lattice.volume
        }

        # Get the crystal system
        crystal_system = spg_analyzer.get_crystal_system()

        # Get the point group
        point_group = spg_analyzer.get_point_group_symbol()
        
        return {
            "success": True,
            "error": None,
            "space_group": space_group,
            "space_group_number": space_group_number,
            "crystal_system": crystal_system,
            "point_group": point_group,
            "chemical_formula": chemical_formula,
            "lattice_parameters": lattice_parameters,
            "num_atoms": len(struct),
            "density": struct.density,
            "elements": [str(el) for el in struct.composition.elements]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error while analyzing crystal structure: {str(e)}\n{traceback.format_exc()}",
            "space_group": None,
            "chemical_formula": None,
            "lattice_parameters": None
        }


def search_materials_project(
    api_key: str,
    search_criteria: Dict[str, Any],
    download_path: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search Materials Project for materials matching the given criteria

    Args:
        api_key: Materials Project API key
        search_criteria: Dictionary of search criteria, supporting the following keys:
            - material_id: str, material ID, e.g. "mp-1234"
            - formula: str, chemical formula, e.g. "TiO2"
            - elements: List[str], element list, e.g. ["Ti", "O"]
            - exclude_elements: List[str], list of elements to exclude
            - band_gap: Tuple[float, float], band gap range (min, max), e.g. (1.0, 3.0)
            - energy_above_hull: Tuple[float, float], energy above hull range (min, max)
            - num_sites: Tuple[int, int], number of atoms range (min, max)
            - spacegroup_number: int, space group number
            - crystal_system: str, crystal system, one of "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"
            - is_gap_direct: bool, whether the band gap is direct
        download_path: Download path; if provided, the structure file is saved there
        limit: Maximum number of results to return

    Returns:
        Dict containing the search results and download status
    """

    try:
        # Build the search parameters
        search_params = {}

        if "material_id" in search_criteria:
            material_id = search_criteria["material_id"]

            if isinstance(material_id, str):
                search_params["material_ids"] = [material_id]
            elif isinstance(material_id, list):
                search_params["material_ids"] = material_id

        # Chemical formula search
        if "formula" in search_criteria:
            search_params["formula"] = search_criteria["formula"]

        # Element composition search
        if "elements" in search_criteria:
            elements = search_criteria["elements"]
            if isinstance(elements, list):
                search_params["elements"] = elements

        # Excluded elements
        if "exclude_elements" in search_criteria:
            exclude_elements = search_criteria["exclude_elements"]
            if isinstance(exclude_elements, list):
                search_params["exclude_elements"] = exclude_elements

        # Band gap range
        if "band_gap" in search_criteria:
            band_gap_range = search_criteria["band_gap"]
            if isinstance(band_gap_range, (tuple, list)) and len(band_gap_range) == 2:
                min_bg, max_bg = band_gap_range
                search_params["band_gap"] = (min_bg, max_bg)
            elif isinstance(band_gap_range, (int, float)):
                # A single value is treated as the lower bound
                search_params["band_gap"] = (band_gap_range, None)

        # Energy above hull range
        if "energy_above_hull" in search_criteria:
            energy_range = search_criteria["energy_above_hull"]
            if isinstance(energy_range,  (tuple, list)) and len(energy_range) == 2:
                search_params["energy_above_hull"] = tuple(energy_range)

        # Number of atoms range
        if "num_sites" in search_criteria:
            nsites_range = search_criteria["num_sites"]
            if isinstance(nsites_range, (tuple, list)) and len(nsites_range) == 2:
                search_params["num_sites"] = tuple(nsites_range)

        # Space group number
        if "spacegroup_number" in search_criteria:
            search_params["spacegroup_number"] = search_criteria["spacegroup_number"]

        # Crystal system
        if "crystal_system" in search_criteria:
            search_params["crystal_system"] = search_criteria["crystal_system"]

        # Direct band gap
        if "is_gap_direct" in search_criteria:
            search_params["is_gap_direct"] = search_criteria["is_gap_direct"]

        search_params["num_chunks"] = 1
        search_params["chunk_size"] = limit
        # Execute the search
        try:
            with MPRester(api_key) as mpr:
                materials_data = mpr.materials.summary.search(
                    **search_params
                )
        except Exception as query_error:
            return {
                "success": False,
                "error": f"Error while searching Materials Project: {str(query_error)}\n{traceback.format_exc()}",
                "materials": [],
                "count": 0,
                "search_criteria": search_criteria
            }
        # Limit the number of results
        if isinstance(materials_data, list):
            materials_data = materials_data[:limit]
        else:
            materials_data = [materials_data]

        if not materials_data:
            return {
                "success": False,
                "error": "No materials found matching the given criteria",
                "materials": [],
                "count": 0,
                "search_criteria": search_criteria
            }

        # Process the search results
        materials_list = []
        for material_data in materials_data:
            try:

                structure: Structure = material_data.structure
                if structure is None:
                    continue

                material_info = {
                    "material_id": material_data.material_id,
                    "formula": structure.composition.reduced_formula,
                    "band_gap": material_data.band_gap,
                    "energy_above_hull": material_data.energy_above_hull,
                    "is_gap_direct": material_data.is_gap_direct,
                }

                # If a download path is provided, save the structure file
                if download_path:
                    os.makedirs(download_path, exist_ok=True)
                    filename = f"{material_data.material_id}_{structure.composition.reduced_formula}.vasp"
                    filepath = os.path.join(download_path, filename)
                    structure.to(filename=filepath, fmt="poscar")
                    material_info["downloaded_file"] = filepath

                materials_list.append(material_info)

            except Exception as material_error:
                print(f"Error while processing material {material_data.material_id}: {str(material_error)}")
                continue

        return {
            "success": True,
            "error": None,
            "materials": materials_list,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error while searching Materials Project: {str(e)}\n{traceback.format_exc()}",
            "materials": [],
            "search_criteria": search_criteria
        }

def create_crystal_structure(
    positions: np.ndarray,
    elements: List[str],
    lattice_vectors: np.ndarray,
    cartesian: bool = False,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a crystal structure

    Args:
        positions: Atomic positions, formatted as [[x1, y1, z1], [x2, y2, z2], ...]
        elements: Element list, e.g. ["Li", "F"]
        lattice_vectors: Lattice vectors, formatted as [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]]
        output_path: Output folder path; if provided, the structure file is saved there

    Returns:
        Dict containing the created structure and related info
    """
    try:
        structure = Structure(lattice=Lattice(lattice_vectors), species=elements, coords=positions, coords_are_cartesian=cartesian)
        structure_id = str(uuid.uuid4())
        structure_name = f"{structure.composition.reduced_formula}_{structure_id}.vasp"
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            poscar = Poscar(structure, sort_structure=True)
            poscar.write_file(filename=f"{output_path}/{structure_name}")
        
        return {
            "success": True,
            "error": None,
            "output_path": f"{output_path}/{structure_name}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error while creating crystal structure:\n {str(e)}",
        }
    

def make_supercell(
    struct_path: str,
    supercell_matrix: List[List[int]],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a supercell structure

    Args:
        struct_input: Structure input, either a file path or a pymatgen Structure object
        supercell_matrix: Supercell matrix, e.g. [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        output_path: Output file path; if provided, the structure file is saved there

    Returns:
        Dict containing the supercell structure and related info
    """

    try:
        # Process the input argument
        if os.path.exists(struct_path):
            fmt = None
            if struct_path.split(".")[-1] in ["poscar", "vasp"]:
                fmt = "poscar"
            elif struct_path.split(".")[-1] in ["cif"]:
                fmt = "cif"
            else:
                fmt = "poscar"
            with open(struct_path, "r") as f:
                struct = Structure.from_str(f.read(), fmt=fmt)
        else:
            return {
                "success": False,
                "error": f"File does not exist: {struct_path}",
                "rotated_structure": None
            }

        # Use pymatgen to create the supercell
        supercell_transform = SupercellTransformation(supercell_matrix)
        supercell_struct = supercell_transform.apply_transformation(struct)

        # If an output path is provided, save the structure file
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = struct_path.replace('.vasp', f'_sc_{supercell_matrix}.vasp')
        supercell_struct.to(filename=output_path, fmt="poscar")

        return {
            "success": True,
            "error": None,
            "original_num_atoms": len(struct),
            "supercell_num_atoms": len(supercell_struct),
            "supercell_matrix": supercell_matrix,
            "output_path": output_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error while creating supercell: {str(e)}\n{traceback.format_exc()}",
            "supercell_structure": None
        }

def scale_structure(
    struct_path: str,
    scale_factors: List[int],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scale a crystal structure

    Args:
        struct_input: Structure input, either a file path or a pymatgen Structure object
        scale_factors: Scale factors, e.g. [2, 2, 1]
        output_path: Output file path; if provided, the structure file is saved there

    Returns:
        Dict containing the scaled structure and related info
    """

    try:
        # Process the input argument
        if os.path.exists(struct_path):
            fmt = None
            if struct_path.split(".")[-1] in ["poscar", "vasp"]:
                fmt = "poscar"
            elif struct_path.split(".")[-1] in ["cif"]:
                fmt = "cif"
            else:
                fmt = "poscar"
            with open(struct_path, "r") as f:
                struct = Structure.from_str(f.read(), fmt=fmt)
        else:
            return {
                "success": False,
                "error": f"File does not exist: {struct_path}",
                "rotated_structure": None
            }

        # Use pymatgen to build the scaled cell
        struct_ase = struct.to_ase_atoms()
        cell = struct_ase.get_cell().array
        cell = np.array(scale_factors) * cell
        struct_ase.set_cell(cell)
        struct = Structure.from_ase_atoms(struct_ase)

        # If an output path is provided, save the structure file
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = struct_path.replace('.vasp', f'_scale_{scale_factors[0]}_{scale_factors[1]}_{scale_factors[2]}.vasp')
        struct.to(filename=output_path, fmt="poscar")

        return {
            "success": True,
            "error": None,
            "num_atoms": len(struct),
            "scale_factors": scale_factors,
            "output_path": output_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error while scaling structure: {str(e)}\n{traceback.format_exc()}",
            "scaled_structure": None
        }


def rotate_structure(
    struct_path: str,
    rotation_axis: List[float],
    angle_degrees: float,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Rotate a crystal structure

    Args:
        struct_input: Structure input, either a file path or a pymatgen Structure object
        rotation_axis: Rotation axis vector, e.g. [0, 0, 1]
        angle_degrees: Rotation angle (degrees)
        output_path: Output file path; if provided, the structure file is saved there

    Returns:
        Dict containing the rotated structure and related info
    """

    try:
        # Process the input argument
        if os.path.exists(struct_path):
            fmt = None
            if struct_path.split(".")[-1] in ["poscar", "vasp"]:
                fmt = "poscar"
            elif struct_path.split(".")[-1] in ["cif"]:
                fmt = "cif"
            else:
                fmt = "poscar"
            with open(struct_path, "r") as f:
                struct = Structure.from_str(f.read(), fmt=fmt)
        else:
            return {
                "success": False,
                "error": f"File does not exist: {struct_path}",
                "rotated_structure": None
            }

        # Use pymatgen to perform the rotation
        rotation_transform = RotationTransformation(rotation_axis, angle_degrees)
        rotated_struct = rotation_transform.apply_transformation(struct)

        # If an output path is provided, save the structure file
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            rotated_struct.to(filename=output_path, fmt="poscar")

        return {
            "success": True,
            "error": None,
            "rotated_structure": rotated_struct,
            "rotation_axis": rotation_axis,
            "angle_degrees": angle_degrees,
            "output_path": output_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error while rotating structure: {str(e)}\n{traceback.format_exc()}",
            "rotated_structure": None
        }


def symmetrize_structure(
    struct_path: str,
    tolerance: float = 0.01,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Symmetrize a crystal structure

    Args:
        struct_input: Structure input, either a file path or a pymatgen Structure object
        tolerance: Symmetry tolerance
        output_path: Output file path; if provided, the structure file is saved there

    Returns:
        Dict containing the symmetrized structure and related info
    """

    try:
        # Process the input argument
        if os.path.exists(struct_path):
            fmt = None
            if struct_path.split(".")[-1] in ["poscar", "vasp"]:
                fmt = "poscar"
            elif struct_path.split(".")[-1] in ["cif"]:
                fmt = "cif"
            else:
                fmt = "poscar"
            with open(struct_path, "r") as f:
                struct = Structure.from_str(f.read(), fmt=fmt)
        else:
            return {
                "success": False,
                "error": f"File does not exist: {struct_path}",
                "symmetrized_structure": None
            }

        # Use pymatgen to perform the symmetrization
        spg_analyzer = SpacegroupAnalyzer(struct, symprec=tolerance)
        symmetrized_struct = spg_analyzer.get_symmetrized_structure()

        # Compare space groups before and after symmetrization
        original_space_group = SpacegroupAnalyzer(struct).get_space_group_symbol()
        symmetrized_space_group = SpacegroupAnalyzer(symmetrized_struct).get_space_group_symbol()

        # If an output path is provided, save the structure file
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            symmetrized_struct.to(filename=output_path, fmt="poscar")
        else:
            output_path = struct_path.replace('.vasp', f'_sym.vasp')
            symmetrized_struct.to(filename=output_path, fmt="poscar")

        return {
            "success": True,
            "error": None,
            "original_space_group": original_space_group,
            "symmetrized_space_group": symmetrized_space_group,
            "original_num_atoms": len(struct),
            "symmetrized_num_atoms": len(symmetrized_struct),
            "tolerance": tolerance,
            "output_path": output_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error while symmetrizing structure: {str(e)}\n{traceback.format_exc()}",
            "symmetrized_structure": None
        }


def convert_structure_format(
    input_path: str,
    output_path: str
) -> Dict[str, Any]:
    """
    Convert a crystal structure file format

    Args:
        input_path: Input file path
        output_path: Output file path

    Returns:
        Dict containing the conversion status and related info
    """

    try:
        # Check whether the input file exists
        if os.path.exists(input_path):
            fmt = None
            if input_path.split(".")[-1] in ["poscar", "vasp"]:
                fmt = "poscar"
            elif input_path.split(".")[-1] in ["cif"]:
                fmt = "cif"
            with open(input_path, "r") as f:
                struct = Structure.from_str(f.read(), fmt=fmt)
        else:
            return {
                "success": False,
                "error": f"File does not exist: {input_path}",
                "converted_structure": None
            }

        # Create the output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the structure
        struct.to(filename=output_path, fmt="poscar")

        return {
            "success": True,
            "error": None,
            "converted_structure": struct,
            "input_path": input_path,
            "output_path": output_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error while converting structure format: {str(e)}\n{traceback.format_exc()}",
            "converted_structure": None
        }
