#!/bin/bash

#SBATCH --job-name=vaspilot
#SBATCH --partition=local
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --output=log
#SBATCH --error=vasp.err

echo "===== VASPilot Job ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "NTASKS=$SLURM_NTASKS"
echo "WORKDIR=$SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR" || exit 1

VASP=/workspace/team/material/vasp.6.4.2/bin/vasp_std

mpirun -np "$SLURM_NTASKS" "$VASP"