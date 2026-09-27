#!/usr/bin/env python3
import sys

def sum_chg(file1_path, file2_path, output_path):
    print(f"Reading {file1_path} and {file2_path}...")
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, open(output_path, 'w') as fout:
        # 1. Copy the VASP header lines exactly
        for _ in range(5):  # System description, scaling factor, 3 lattice vectors
            fout.write(f1.readline())
            f2.readline()
            
        # Species types and counts (VASP 5 format uses 2 lines here)
        types_line = f1.readline()
        fout.write(types_line)
        f2.readline()
        
        counts_line = f1.readline()
        fout.write(counts_line)
        f2.readline()
        
        # Calculate total number of atoms from counts line
        num_atoms = sum(int(x) for x in counts_line.split())
        
        # Coordinate type line (Direct/Cartesian)
        fout.write(f1.readline())
        f2.readline()
        
        # Copy atomic positions
        for _ in range(num_atoms):
            fout.write(f1.readline())
            f2.readline()
            
        # Copy the blank line before grid dimensions
        fout.write(f1.readline())
        f2.readline()
        
        # Read and copy grid dimensions line
        grid_line = f1.readline()
        fout.write(grid_line)
        f2.readline()
        
        grid_dims = [int(x) for x in grid_line.split()]
        total_points = grid_dims[0] * grid_dims[1] * grid_dims[2]
        print(f"Grid detected: {grid_dims[0]}x{grid_dims[1]}x{grid_dims[2]} ({total_points} total points)")
        
        # 2. Parse and sum numerical values safely with proper spacing
        print("Processing charge density arrays...")
        val1_gen = (float(val) for line in f1 for val in line.split())
        val2_gen = (float(val) for line in f2 for val in line.split())
        
        out_line = []
        count = 0
        
        for _ in range(total_points):
            try:
                v1 = next(val1_gen)
                v2 = next(val2_gen)
                total = v1 + v2
                # Use standard format with an explicitly padded leading space
                out_line.append(f" {total:17.11E}")
                count += 1
                
                if len(out_line) == 5:
                    fout.write("".join(out_line) + "\n")
                    out_line = []
            except StopIteration:
                break
                
        if out_line:
            fout.write("".join(out_line) + "\n")
            
        print(f"Successfully summed {count} grid values.")
        
        # 3. Stream remaining data (PAW occupancies at the bottom of VASP 5 files)
        print("Copying remaining structural metadata...")
        # Fast-forward f1 to the line right after grid data block
        # (Since we consumed total_points items from the generator, we read the remainder)
        for line in f1:
            fout.write(line)
            
    print(f"Done! Saved file as: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chgsum.py AECCAR0 AECCAR2 [OUTPUT]")
        sys.exit(1)
        
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "CHGCAR_sum"
    
    sum_chg(file1, file2, output)