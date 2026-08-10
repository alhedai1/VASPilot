import os
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from pymatgen.core import Structure, Element, Lattice
from mp_api.client import MPRester
from pymatgen.transformations.advanced_transformations import SupercellTransformation
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
import uuid
from pymatgen.io.vasp import Poscar

def analyze_crystal_structure(struct_input: Union[str, Structure]) -> Dict[str, Any]:
    """
    分析晶体结构的空间群和化学表达式
    
    参数:
        struct_input: 结构输入，可以是文件路径或pymatgen Structure对象
        
    返回:
        Dict包含空间群信息、化学表达式、晶格参数等
    """
    
    try:
        # 处理输入参数
        if isinstance(struct_input, str):
            # 如果是文件路径
            if os.path.exists(struct_input):
                struct = Structure.from_file(struct_input)
            else:
                return {
                    "success": False,
                    "error": f"文件不存在: {struct_input}",
                    "space_group": None,
                    "chemical_formula": None,
                    "lattice_parameters": None
                }
        elif isinstance(struct_input, Structure):
            struct = struct_input
        else:
            return {
                "success": False,
                "error": "不支持的输入类型，请提供文件路径或pymatgen Structure对象",
                "space_group": None,
                "chemical_formula": None,
                "lattice_parameters": None
            }
        
        # 使用pymatgen分析空间群
        spg_analyzer = SpacegroupAnalyzer(struct)
        space_group = spg_analyzer.get_space_group_symbol()
        space_group_number = spg_analyzer.get_space_group_number()
        
        # 获取化学表达式
        chemical_formula = struct.composition.reduced_formula
        
        # 获取晶格参数
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
        
        # 获取晶体系统
        crystal_system = spg_analyzer.get_crystal_system()
        
        # 获取点群
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
            "error": f"分析晶体结构时出错: {str(e)}\n{traceback.format_exc()}",
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
    从Materials Project根据条件搜索材料
    
    参数:
        api_key: Materials Project API密钥
        search_criteria: 搜索条件字典，支持以下键值：
            - material_id: str, 材料ID，例如 "mp-1234"
            - formula: str, 化学式，例如 "TiO2"
            - elements: List[str], 元素列表，例如 ["Ti", "O"]
            - exclude_elements: List[str], 排除的元素列表
            - band_gap: Tuple[float, float], 带隙范围 (min, max)，例如 (1.0, 3.0)
            - energy_above_hull: Tuple[float, float], 形成能范围 (min, max)
            - num_sites: Tuple[int, int], 原子数范围 (min, max)
            - spacegroup_number: int, 空间群编号
            - crystal_system: str, 晶系, "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"中的一个
            - is_gap_direct: bool, 是否为直接带隙
        download_path: 下载路径，如果提供则保存结构文件
        limit: 返回结果的最大数量
        
    返回:
        Dict包含搜索结果和下载状态
    """
    
    try:
        # 构建搜索条件
        search_params = {}

        if "material_id" in search_criteria:
            material_id = search_criteria["material_id"]

            if isinstance(material_id, str):
                search_params["material_ids"] = [material_id]
            elif isinstance(material_id, list):
                search_params["material_ids"] = material_id
        
        # 化学式搜索
        if "formula" in search_criteria:
            search_params["formula"] = search_criteria["formula"]
        
        # 元素组成搜索
        if "elements" in search_criteria:
            elements = search_criteria["elements"]
            if isinstance(elements, list):
                search_params["elements"] = elements
        
        # 排除元素
        if "exclude_elements" in search_criteria:
            exclude_elements = search_criteria["exclude_elements"]
            if isinstance(exclude_elements, list):
                search_params["exclude_elements"] = exclude_elements
        
        # 带隙范围
        if "band_gap" in search_criteria:
            band_gap_range = search_criteria["band_gap"]
            if isinstance(band_gap_range, (tuple, list)) and len(band_gap_range) == 2:
                min_bg, max_bg = band_gap_range
                search_params["band_gap"] = (min_bg, max_bg)
            elif isinstance(band_gap_range, (int, float)):
                # 单值视为下限
                search_params["band_gap"] = (band_gap_range, None)
        
        # 形成能范围
        if "energy_above_hull" in search_criteria:
            energy_range = search_criteria["energy_above_hull"]
            if isinstance(energy_range,  (tuple, list)) and len(energy_range) == 2:
                search_params["energy_above_hull"] = tuple(energy_range)
        
        # 原子数范围
        if "num_sites" in search_criteria:
            nsites_range = search_criteria["num_sites"]
            if isinstance(nsites_range, (tuple, list)) and len(nsites_range) == 2:
                search_params["num_sites"] = tuple(nsites_range)
        
        # 空间群编号
        if "spacegroup_number" in search_criteria:
            search_params["spacegroup_number"] = search_criteria["spacegroup_number"]
        
        # 晶系
        if "crystal_system" in search_criteria:
            search_params["crystal_system"] = search_criteria["crystal_system"]
        
        # 直接带隙
        if "is_gap_direct" in search_criteria:
            search_params["is_gap_direct"] = search_criteria["is_gap_direct"]
        
        search_params["num_chunks"] = 1
        search_params["chunk_size"] = limit
        # 执行搜索
        try:
            with MPRester(api_key) as mpr:
                materials_data = mpr.materials.summary.search(
                    **search_params
                )
        except Exception as query_error:
            return {
                "success": False,
                "error": f"搜索Materials Project时出错: {str(query_error)}\n{traceback.format_exc()}",
                "materials": [],
                "count": 0,
                "search_criteria": search_criteria
            }
        # 限制结果数量
        if isinstance(materials_data, list):
            materials_data = materials_data[:limit]
        else:
            materials_data = [materials_data]
        
        if not materials_data:
            return {
                "success": False,
                "error": "未找到符合条件的材料",
                "materials": [],
                "count": 0,
                "search_criteria": search_criteria
            }

        # 处理搜索结果
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
                
                # 如果提供了下载路径，保存结构文件
                if download_path:
                    os.makedirs(download_path, exist_ok=True)
                    filename = f"{material_data.material_id}_{structure.composition.reduced_formula}.vasp"
                    filepath = os.path.join(download_path, filename)
                    structure.to(filename=filepath, fmt="poscar")
                    material_info["downloaded_file"] = filepath
                
                materials_list.append(material_info)
                
            except Exception as material_error:
                print(f"处理材料 {material_data.material_id} 时出错: {str(material_error)}")
                continue
        
        return {
            "success": True,
            "error": None,
            "materials": materials_list,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"搜索Materials Project时出错: {str(e)}\n{traceback.format_exc()}",
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
    创建晶体结构
    
    参数:
        positions: 原子位置，格式为[[x1, y1, z1], [x2, y2, z2], ...]
        elements: 元素列表，例如 ["Li", "F"]
        lattice_vectors: 晶格向量，格式为[[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]]
        output_path: 输出文件夹路径，如果提供则保存结构文件
        
    返回:
        Dict包含创建的结构和相关信息
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
            "error": f"创建晶体结构时出错:\n {str(e)}",
        }
    

def make_supercell(
    struct_path: str,
    supercell_matrix: List[List[int]],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建超胞结构
    
    参数:
        struct_input: 结构输入，可以是文件路径或pymatgen Structure对象
        supercell_matrix: 超胞矩阵，例如 [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        output_path: 输出文件路径，如果提供则保存结构文件
        
    返回:
        Dict包含超胞结构和相关信息
    """
    
    try:
        # 处理输入参数
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
                "error": f"文件不存在: {struct_path}",
                "rotated_structure": None
            }
        
        # 使用pymatgen创建超胞
        supercell_transform = SupercellTransformation(supercell_matrix)
        supercell_struct = supercell_transform.apply_transformation(struct)
        
        # 如果提供了输出路径，保存结构文件
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
            "error": f"创建超胞时出错: {str(e)}\n{traceback.format_exc()}",
            "supercell_structure": None
        }

def scale_structure(
    struct_path: str,
    scale_factors: List[int],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    缩放晶体结构
    
    参数:
        struct_input: 结构输入，可以是文件路径或pymatgen Structure对象
        scale_factors: 缩放因子，例如 [2, 2, 1]
        output_path: 输出文件路径，如果提供则保存结构文件
        
    返回:
        Dict包含缩放后的结构和相关信息
    """
    
    try:
        # 处理输入参数
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
                "error": f"文件不存在: {struct_path}",
                "rotated_structure": None
            }
        
        # 使用pymatgen创建超胞
        struct_ase = struct.to_ase_atoms()
        cell = struct_ase.get_cell().array
        cell = np.array(scale_factors) * cell
        struct_ase.set_cell(cell)
        struct = Structure.from_ase_atoms(struct_ase)
        
        # 如果提供了输出路径，保存结构文件
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
            "error": f"缩放结构时出错: {str(e)}\n{traceback.format_exc()}",
            "scaled_structure": None
        }


def rotate_structure(
    struct_path: str,
    rotation_axis: List[float],
    angle_degrees: float,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    旋转晶体结构
    
    参数:
        struct_input: 结构输入，可以是文件路径或pymatgen Structure对象
        rotation_axis: 旋转轴向量，例如 [0, 0, 1]
        angle_degrees: 旋转角度（度）
        output_path: 输出文件路径，如果提供则保存结构文件
        
    返回:
        Dict包含旋转后的结构和相关信息
    """
    
    try:
        # 处理输入参数
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
                "error": f"文件不存在: {struct_path}",
                "rotated_structure": None
            }
        
        # 使用pymatgen进行旋转
        rotation_transform = RotationTransformation(rotation_axis, angle_degrees)
        rotated_struct = rotation_transform.apply_transformation(struct)
        
        # 如果提供了输出路径，保存结构文件
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
            "error": f"旋转结构时出错: {str(e)}\n{traceback.format_exc()}",
            "rotated_structure": None
        }


def symmetrize_structure(
    struct_path: str,
    tolerance: float = 0.01,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    对称化晶体结构
    
    参数:
        struct_input: 结构输入，可以是文件路径或pymatgen Structure对象
        tolerance: 对称性容忍度
        output_path: 输出文件路径，如果提供则保存结构文件
        
    返回:
        Dict包含对称化后的结构和相关信息
    """
    
    try:
        # 处理输入参数
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
                "error": f"文件不存在: {struct_path}",
                "symmetrized_structure": None
            }
        
        # 使用pymatgen进行对称化
        spg_analyzer = SpacegroupAnalyzer(struct, symprec=tolerance)
        symmetrized_struct = spg_analyzer.get_symmetrized_structure()
        
        # 获取对称化前后的比较信息
        original_space_group = SpacegroupAnalyzer(struct).get_space_group_symbol()
        symmetrized_space_group = SpacegroupAnalyzer(symmetrized_struct).get_space_group_symbol()
        
        # 如果提供了输出路径，保存结构文件
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
            "error": f"对称化结构时出错: {str(e)}\n{traceback.format_exc()}",
            "symmetrized_structure": None
        }


def convert_structure_format(
    input_path: str,
    output_path: str
) -> Dict[str, Any]:
    """
    转换晶体结构文件格式
    
    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        
    返回:
        Dict包含转换状态和相关信息
    """
    
    try:
        # 检查输入文件是否存在
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
                "error": f"文件不存在: {input_path}",
                "converted_structure": None
            }
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存结构
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
            "error": f"转换结构格式时出错: {str(e)}\n{traceback.format_exc()}",
            "converted_structure": None
        }
