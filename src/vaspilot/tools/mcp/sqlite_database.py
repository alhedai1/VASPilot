import sqlite3
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class VaspCalculationDB:
    """SQLite database manager for VASP calculation records"""

    def __init__(self, db_path: str):
        """
        Initialize the database connection

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize the database table schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calculations (
                    calculation_id TEXT PRIMARY KEY,
                    slurm_id TEXT,
                    success BOOLEAN,
                    error TEXT,
                    status TEXT,
                    calculate_path TEXT,
                    calc_type TEXT,

                    -- relaxation-related fields
                    total_energy REAL,
                    max_force REAL,
                    ionic_steps INTEGER,

                    -- scf/nscf-related fields
                    efermi REAL,
                    is_metal BOOLEAN,

                    -- control parameters
                    soc BOOLEAN,
                    restart_id TEXT,
                    kpath TEXT,
                    n_kpoints INTEGER,

                    -- timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- BLOB storage for complex objects
                    structure_blob BLOB,
                    band_structure_blob BLOB,
                    dos_blob BLOB,
                    eigenvalues_blob BLOB,
                    band_gap_blob BLOB,
                    stress_blob BLOB,
                    incar_tags_blob BLOB,
                    cbm_blob BLOB,
                    vbm_blob BLOB
                )
            """)

            # Create indexes to improve query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calc_type ON calculations(calc_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON calculations(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_restart_id ON calculations(restart_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON calculations(created_at)")

    def write_record(self, calculation_id: str, data: dict):
        """
        Write a calculation record

        Args:
            calculation_id: Calculation ID
            data: Calculation data dictionary
        """
        # Extract simple fields
        simple_fields = {
            'calculation_id': calculation_id,
            'slurm_id': data.get('slurm_id'),
            'success': data.get('success'),
            'error': data.get('error'),
            'status': data.get('status'),
            'calculate_path': data.get('calculate_path'),
            'calc_type': data.get('calc_type'),
            'total_energy': data.get('total_energy'),
            'max_force': data.get('max_force'),
            'ionic_steps': data.get('ionic_steps'),
            'efermi': data.get('efermi'),
            'is_metal': data.get('is_metal'),
            'soc': data.get('soc'),
            'restart_id': data.get('restart_id'),
            'kpath': data.get('kpath'),
            'n_kpoints': data.get('n_kpoints'),
        }

        # Serialize complex objects
        blob_fields = {}
        complex_field_mapping = {
            'structure': 'structure_blob',
            'band_structure': 'band_structure_blob',
            'dos': 'dos_blob',
            'eigenvalues': 'eigenvalues_blob',
            'eigen_values': 'eigenvalues_blob',  # supports alternate naming
            'band_gap': 'band_gap_blob',
            'stress': 'stress_blob',
            'incar_tags': 'incar_tags_blob',
            'cbm': 'cbm_blob',
            'vbm': 'vbm_blob'
        }

        for data_key, blob_key in complex_field_mapping.items():
            if data_key in data and data[data_key] is not None:
                try:
                    blob_fields[blob_key] = pickle.dumps(data[data_key])
                except Exception as e:
                    logging.warning(f"Failed to serialize {data_key}: {e}")
                    blob_fields[blob_key] = None
            else:
                blob_fields[blob_key] = None

        # Merge all fields
        all_fields = {**simple_fields, **blob_fields}

        with sqlite3.connect(self.db_path) as conn:
            # Use INSERT OR REPLACE to support updates
            placeholders = ', '.join(['?' for _ in all_fields])
            columns = ', '.join(all_fields.keys())

            conn.execute(f"""
                INSERT OR REPLACE INTO calculations ({columns})
                VALUES ({placeholders})
            """, list(all_fields.values()))

            # Update the timestamp
            conn.execute(
                "UPDATE calculations SET updated_at = CURRENT_TIMESTAMP WHERE calculation_id = ?",
                (calculation_id,)
            )

    def read_record(self, calculation_id: str) -> Optional[Dict[str, Any]]:
        """
        Read a calculation record

        Args:
            calculation_id: Calculation ID

        Returns:
            Calculation data dictionary, or None if it does not exist
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # allow accessing results by column name
            cursor = conn.execute(
                "SELECT * FROM calculations WHERE calculation_id = ?",
                (calculation_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Convert to dict and deserialize complex objects
            data = dict(row)

            # Deserialize BLOB fields
            blob_field_mapping = {
                'structure_blob': 'structure',
                'band_structure_blob': 'band_structure',
                'dos_blob': 'dos',
                'eigenvalues_blob': 'eigenvalues',
                'band_gap_blob': 'band_gap',
                'stress_blob': 'stress',
                'incar_tags_blob': 'incar_tags',
                'cbm_blob': 'cbm',
                'vbm_blob': 'vbm'
            }

            for blob_key, data_key in blob_field_mapping.items():
                if data[blob_key] is not None:
                    try:
                        data[data_key] = pickle.loads(data[blob_key])
                    except Exception as e:
                        logging.warning(f"Failed to deserialize {data_key}: {e}")
                        data[data_key] = None
                else:
                    data[data_key] = None
                # Drop the blob field to preserve interface compatibility
                del data[blob_key]

            # Drop timestamp fields to preserve interface compatibility
            data.pop('created_at', None)
            data.pop('updated_at', None)

            return data

    def list_calculations(self,
                         calc_type: Optional[str] = None,
                         status: Optional[str] = None,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List calculation records

        Args:
            calc_type: Filter by calculation type
            status: Filter by status
            limit: Limit on the number of records returned

        Returns:
            List of calculation records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT calculation_id, calc_type, status, total_energy, efermi, created_at FROM calculations"
            params = []
            conditions = []

            if calc_type:
                conditions.append("calc_type = ?")
                params.append(calc_type)

            if status:
                conditions.append("status = ?")
                params.append(status)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def delete_record(self, calculation_id: str) -> bool:
        """
        Delete a calculation record

        Args:
            calculation_id: Calculation ID

        Returns:
            Whether the deletion succeeded
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM calculations WHERE calculation_id = ?",
                (calculation_id,)
            )
            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Total number of records
            cursor = conn.execute("SELECT COUNT(*) FROM calculations")
            stats['total_calculations'] = cursor.fetchone()[0]

            # Statistics by calculation type
            cursor = conn.execute("SELECT calc_type, COUNT(*) FROM calculations GROUP BY calc_type")
            stats['by_calc_type'] = dict(cursor.fetchall())

            # Statistics by status
            cursor = conn.execute("SELECT status, COUNT(*) FROM calculations GROUP BY status")
            stats['by_status'] = dict(cursor.fetchall())

            return stats
