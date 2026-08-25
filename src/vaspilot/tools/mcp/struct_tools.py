import os
import hashlib
import json
import traceback
from typing import Dict, Any, List, Optional, Union, Literal
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
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


def retrieve_bulk_parent(
    api_key: str,
    download_path: str,
    formula: Optional[str] = None,
    material_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve one deterministic bulk parent structure from Materials Project.

    An explicit Materials Project ID takes precedence. Otherwise, the entry with
    the lowest energy above hull is selected for the requested formula, followed
    by formation energy per atom and material ID as deterministic tie-breakers.
    """
    if not material_id and not formula:
        return {
            "success": False,
            "error": "Either material_id or formula must be provided",
        }

    try:
        search_params = (
            {"material_ids": [material_id]}
            if material_id
            else {"formula": formula}
        )
        with MPRester(api_key) as mpr:
            materials = mpr.materials.summary.search(**search_params)

        if not materials:
            query = f"material_id={material_id}" if material_id else f"formula={formula}"
            return {
                "success": False,
                "error": f"No Materials Project bulk structure found for {query}",
            }

        def selection_key(material: Any) -> tuple:
            hull = getattr(material, "energy_above_hull", None)
            formation = getattr(material, "formation_energy_per_atom", None)
            return (
                float("inf") if hull is None else float(hull),
                float("inf") if formation is None else float(formation),
                str(material.material_id),
            )

        selected = materials[0] if material_id else min(materials, key=selection_key)
        structure: Structure = selected.structure
        if structure is None:
            return {
                "success": False,
                "error": f"Materials Project entry {selected.material_id} has no structure",
            }

        os.makedirs(download_path, exist_ok=True)
        selected_id = str(selected.material_id)
        selected_formula = structure.composition.reduced_formula
        filepath = os.path.abspath(
            os.path.join(download_path, f"{selected_id}_{selected_formula}.vasp")
        )
        structure.to(filename=filepath, fmt="poscar")

        hull = getattr(selected, "energy_above_hull", None)
        formation = getattr(selected, "formation_energy_per_atom", None)
        return {
            "success": True,
            "error": None,
            "structure_kind": "bulk_parent",
            "material_id": selected_id,
            "formula": selected_formula,
            "energy_above_hull_eV_per_atom": None if hull is None else float(hull),
            "formation_energy_eV_per_atom": (
                None if formation is None else float(formation)
            ),
            "selection_method": (
                "explicit_material_id"
                if material_id
                else "lowest_energy_above_hull_then_formation_energy"
            ),
            "structure_path": filepath,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error while retrieving bulk parent: {str(e)}\n{traceback.format_exc()}",
        }


def build_surface(
    bulk_structure_path: str,
    miller_index: List[int],
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 15.0,
    lateral_supercell: List[int] = [3, 3],
    fixed_bottom_layers: int = 2,
    orthogonalize_c: bool = False,
    termination_policy: Literal["lowest_shift", "index"] = "lowest_shift",
    termination_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a validated, reproducible slab from a bulk structure."""
    layer_tolerance = 0.5

    def fail(message: str) -> Dict[str, Any]:
        return {"success": False, "error": message}

    if not os.path.isfile(bulk_structure_path):
        return fail(f"Bulk structure file does not exist: {bulk_structure_path}")
    if (
        not isinstance(miller_index, (list, tuple))
        or len(miller_index) != 3
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in miller_index)
        or not any(miller_index)
    ):
        return fail("miller_index must contain three integers and cannot be [0, 0, 0]")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
               for value in (min_slab_size, min_vacuum_size)):
        return fail("min_slab_size and min_vacuum_size must be positive numbers")
    if (not isinstance(lateral_supercell, (list, tuple)) or len(lateral_supercell) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                       for value in lateral_supercell)):
        return fail("lateral_supercell must contain two positive integers")
    if not isinstance(fixed_bottom_layers, int) or isinstance(fixed_bottom_layers, bool) or fixed_bottom_layers < 0:
        return fail("fixed_bottom_layers must be a non-negative integer")
    if not isinstance(orthogonalize_c, bool):
        return fail("orthogonalize_c must be a boolean")
    if termination_policy not in {"lowest_shift", "index"}:
        return fail("termination_policy must be 'lowest_shift' or 'index'")
    if termination_policy == "index":
        if not isinstance(termination_index, int) or isinstance(termination_index, bool) or termination_index < 0:
            return fail("termination_index must be a non-negative integer for policy 'index'")
    elif termination_index is not None:
        return fail("termination_index may only be used with policy 'index'")

    try:
        miller = tuple(int(value) for value in miller_index)
        lateral = [int(value) for value in lateral_supercell]
        request = {
            "miller_index": list(miller),
            "min_slab_size_A": float(min_slab_size),
            "min_vacuum_size_A": float(min_vacuum_size),
            "lateral_supercell": lateral,
            "fixed_bottom_layers": fixed_bottom_layers,
            "orthogonalize_c": orthogonalize_c,
            "termination_policy": termination_policy,
            "termination_index": termination_index,
        }
        with open(bulk_structure_path, "rb") as stream:
            input_sha256 = hashlib.sha256(stream.read()).hexdigest()
        bulk = Structure.from_file(bulk_structure_path)
        conventional_bulk = SpacegroupAnalyzer(
            bulk, symprec=0.01
        ).get_conventional_standard_structure()

        generator = SlabGenerator(
            initial_structure=conventional_bulk,
            miller_index=miller,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=True,
            in_unit_planes=False,
            primitive=True,
            reorient_lattice=True,
        )
        terminations = generator.get_slabs(symmetrize=False)
        if not terminations:
            return {
                "success": False,
                "error": f"No slab termination generated for Miller index {miller}",
            }

        terminations = sorted(
            terminations,
            key=lambda slab: (
                round(float(getattr(slab, "shift", 0.0)), 12),
                len(slab),
                slab.composition.reduced_formula,
            ),
        )
        selected_index = 0 if termination_policy == "lowest_shift" else termination_index
        if selected_index >= len(terminations):
            return fail(
                f"termination_index {selected_index} is out of range for "
                f"{len(terminations)} generated terminations"
            )
        slab = terminations[selected_index].copy()
        selected_shift = float(getattr(slab, "shift", 0.0))
        slab.make_supercell([*lateral, 1])

        a_vector, b_vector, c_vector = slab.lattice.matrix
        surface_normal = np.cross(a_vector, b_vector)
        surface_area = float(np.linalg.norm(surface_normal))
        surface_normal /= surface_area
        cell_height = float(abs(np.dot(c_vector, surface_normal)))
        if np.dot(c_vector, surface_normal) < 0:
            surface_normal *= -1
        if orthogonalize_c:
            slab = Structure(
                Lattice([a_vector, b_vector, surface_normal * cell_height]),
                slab.species,
                slab.cart_coords,
                coords_are_cartesian=True,
                site_properties=slab.site_properties,
            )
            a_vector, b_vector, c_vector = slab.lattice.matrix
        heights = np.dot(slab.cart_coords, surface_normal)

        layers: List[List[float]] = []
        for height in sorted(float(value) for value in heights):
            if not layers or height - layers[-1][-1] > layer_tolerance:
                layers.append([height])
            else:
                layers[-1].append(height)
        layer_heights = [float(np.mean(layer)) for layer in layers]
        atom_layer_indices = [
            min(range(len(layer_heights)), key=lambda index: abs(height - layer_heights[index]))
            for height in heights
        ]
        fixed_layer_count = min(fixed_bottom_layers, len(layers))
        fixed_atom_indices = [index for index, layer in enumerate(atom_layer_indices)
                              if layer < fixed_layer_count]
        fixed_set = set(fixed_atom_indices)
        slab.add_site_property(
            "selective_dynamics",
            [
                [False, False, False] if index in fixed_set else [True, True, True]
                for index in range(len(slab))
            ],
        )

        provenance = {"input_sha256": input_sha256, "request": request,
                      "selected_termination_index": selected_index,
                      "selected_termination_shift": selected_shift}
        artifact_id = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        miller_label = "_".join(str(value).replace("-", "m") for value in miller)
        input_stem = os.path.splitext(os.path.basename(bulk_structure_path))[0]
        output_path = os.path.abspath(
            os.path.join(
                os.path.dirname(bulk_structure_path),
                f"{input_stem}_hkl-{miller_label}_sc-{lateral[0]}x{lateral[1]}_t-{selected_index}_{artifact_id}.vasp",
            )
        )
        Poscar(slab, sort_structure=False).write_file(output_path)

        slab_thickness = float(max(heights) - min(heights))
        return {
            "success": True,
            "error": None,
            "structure_kind": "surface_slab",
            "bulk_structure_path": os.path.abspath(bulk_structure_path),
            "slab_structure_path": output_path,
            "miller_index": list(miller),
            "artifact_id": artifact_id,
            "input_structure_sha256": input_sha256,
            "num_atoms": len(slab),
            "num_atomic_layers": len(layer_heights),
            "atomic_layer_indices": atom_layer_indices,
            "atomic_layer_heights_A": layer_heights,
            "fixed_atom_indices": fixed_atom_indices,
            "fixed_indexing": "zero_based",
            "surface_area_A2": surface_area,
            "surface_normal_cartesian": surface_normal.tolist(),
            "cell_height_along_normal_A": cell_height,
            "estimated_slab_thickness_A": slab_thickness,
            "estimated_vacuum_thickness_A": cell_height - slab_thickness,
            "surface_metadata": {
                "bulk_standardization": "conventional_standard_structure",
                "construction_parameters": request,
                "minimum_slab_thickness_A": min_slab_size,
                "minimum_vacuum_thickness_A": min_vacuum_size,
                "lateral_supercell": lateral,
                "fixed_bottom_layers": fixed_layer_count,
                "orthogonalized_c": orthogonalize_c,
                "termination_count": len(terminations),
                "selected_termination_index": selected_index,
                "selected_termination_shift": selected_shift,
                "termination_selection": termination_policy,
                "termination_energy_ranked": False,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error while building surface: {str(e)}\n{traceback.format_exc()}",
        }


def build_adsorbate(
    adsorbate: str,
    output_directory: str,
) -> Dict[str, Any]:
    """Build a validated adsorbate from a small curated ASE-backed registry."""
    registry = {
        "H": {"binding_element": "H", "charge": 0, "multiplicity": 2},
        "O": {"binding_element": "O", "charge": 0, "multiplicity": 3},
        "N": {"binding_element": "N", "charge": 0, "multiplicity": 4},
        "CO": {"binding_element": "C", "charge": 0, "multiplicity": 1},
        "OH": {"binding_element": "O", "charge": 0, "multiplicity": 2},
        "H2O": {"binding_element": "O", "charge": 0, "multiplicity": 1},
        "CO2": {"binding_element": "C", "charge": 0, "multiplicity": 1},
    }

    if not isinstance(adsorbate, str) or not adsorbate.strip():
        return {"success": False, "error": "adsorbate must be a non-empty string"}

    adsorbate_name = adsorbate.strip().upper()
    if adsorbate_name not in registry:
        return {
            "success": False,
            "error": (
                f"Unsupported adsorbate '{adsorbate}'. Supported adsorbates: "
                f"{', '.join(registry)}"
            ),
        }

    try:
        from ase import Atoms
        from ase.build import molecule
        from ase.io import write

        atoms = (
            Atoms(adsorbate_name)
            if adsorbate_name in {"H", "O", "N"}
            else molecule(adsorbate_name)
        )
        metadata = registry[adsorbate_name]
        symbols = atoms.get_chemical_symbols()
        binding_element = metadata["binding_element"]
        binding_atom_index = symbols.index(binding_element)

        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.abspath(
            os.path.join(output_directory, f"adsorbate_{adsorbate_name}.xyz")
        )
        write(output_path, atoms, format="xyz")

        return {
            "success": True,
            "error": None,
            "structure_kind": "adsorbate",
            "adsorbate": adsorbate_name,
            "adsorbate_structure_path": output_path,
            "chemical_symbols": symbols,
            "num_atoms": len(atoms),
            "binding_atom_indices": [binding_atom_index],
            "binding_indexing": "zero_based",
            "binding_atom_elements": [binding_element],
            "charge": metadata["charge"],
            "multiplicity": metadata["multiplicity"],
            "construction_source": "ASE molecule library",
            "orientation_policy": "deferred_to_candidate_generation",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error while building adsorbate: {str(e)}\n{traceback.format_exc()}",
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
