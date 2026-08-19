import io
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
import base64
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')  # use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element, Lattice
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.dos import CompleteDos


def safe_execute_plot_code(plot_code: str, data: Dict[str, Any], work_dir: str) -> tuple[bool, str, Optional[str]]:
    """Safely execute plotting code"""
    try:
        # Redirect output
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Create the execution environment, including the data variable and required libraries
        exec_globals = {
            'data': data,
            'plt': plt,
            'np': np,
            'pd': pd,
            'Structure': Structure,
            'Element': Element,
            'Lattice': Lattice,
            'BandStructure': BandStructure,
            'CompleteDos': CompleteDos,
            'Vasprun': Vasprun,
            '__builtins__': __builtins__
        }
        
        # Import commonly used pymatgen modules
        try:
            from pymatgen.electronic_structure.core import Spin
            from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
            exec_globals['Spin'] = Spin
            exec_globals['BSPlotter'] = BSPlotter
            exec_globals['DosPlotter'] = DosPlotter
        except ImportError:
            pass  # Continue execution if some modules are unavailable

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(plot_code, exec_globals)

        # Generate a unique image filename
        plot_id = str(uuid.uuid4())
        plot_filename = f"plot_{plot_id}.png"
        plot_path = Path(work_dir) / plot_filename

        # Save the image
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # close the figure to free memory

        # Read the image and convert it to base64
        with open(plot_path, 'rb') as f:
            img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

        return True, str(plot_path), img_base64

    except Exception as e:
        plt.close()  # ensure the figure is closed even on error
        error_msg = f"Error while executing plotting code: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg, None