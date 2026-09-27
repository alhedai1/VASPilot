from pymatgen.core import Structure
import numpy as np

# =========================
# SETTINGS
# =========================
SLAB_FILE = "/workspace/team/material/VASPilot/examples/1.Basic/mcp/downloads/mp-2646977_Pd_hkl-1_1_1_sc-3x3_t-0_a96b270ebd49b3e6.vasp"

# If you know the cubic lattice constant of the SAME Pd bulk structure
# used to generate the slab, put it here.
# Otherwise leave as None.
PD_BULK_A = None
# Example:
# PD_BULK_A = 3.89


# =========================
# LOAD SLAB
# =========================
slab = Structure.from_file(SLAB_FILE)

print("=" * 60)
print("BASIC INFORMATION")
print("=" * 60)

print("Composition:", slab.composition)
print("Number of atoms:", len(slab))

print("\nLattice lengths [Å]:")
print(f"a = {slab.lattice.a:.4f}")
print(f"b = {slab.lattice.b:.4f}")
print(f"c = {slab.lattice.c:.4f}")

print("\nLattice angles [deg]:")
print(f"alpha = {slab.lattice.alpha:.4f}")
print(f"beta  = {slab.lattice.beta:.4f}")
print(f"gamma = {slab.lattice.gamma:.4f}")

print("\nLattice matrix:")
print(slab.lattice.matrix)


# =========================
# 1. COMPOSITION CHECK
# =========================
print("\n" + "=" * 60)
print("1. COMPOSITION CHECK")
print("=" * 60)

elements = {str(el) for el in slab.composition.elements}

if elements == {"Pd"}:
    print("PASS: slab contains only Pd.")
else:
    print("WARNING: slab contains elements:", elements)


# =========================
# 2. NEAREST-NEIGHBOR DISTANCES
# =========================
print("\n" + "=" * 60)
print("2. Pd-Pd NEAREST-NEIGHBOR DISTANCES")
print("=" * 60)

distance_matrix = slab.distance_matrix

nn_distances = []

for i in range(len(slab)):
    row = distance_matrix[i]
    nonzero = row[row > 1e-6]
    nn_distances.append(nonzero.min())

nn_distances = np.array(nn_distances)

print(f"Minimum NN distance : {nn_distances.min():.4f} Å")
print(f"Mean NN distance    : {nn_distances.mean():.4f} Å")
print(f"Maximum NN distance : {nn_distances.max():.4f} Å")

if PD_BULK_A is not None:
    expected_nn = PD_BULK_A / np.sqrt(2)

    print(f"\nExpected FCC Pd NN distance from bulk:")
    print(f"a_bulk / sqrt(2) = {expected_nn:.4f} Å")

    relative_error = abs(nn_distances.mean() - expected_nn) / expected_nn * 100

    print(f"Difference from expected = {relative_error:.2f}%")

    if relative_error < 5:
        print("PASS: nearest-neighbor distance is consistent with FCC Pd.")
    else:
        print("WARNING: nearest-neighbor distance differs noticeably from bulk Pd.")


# =========================
# 3. IDENTIFY ATOMIC LAYERS
# =========================
print("\n" + "=" * 60)
print("3. ATOMIC LAYERS")
print("=" * 60)

# Assumes slab surface is approximately parallel to the a-b plane,
# i.e. c is the slab/vacuum direction.

z_cart = np.array([site.coords[2] for site in slab])

# Sort z values
z_sorted = np.sort(z_cart)

# Atoms whose z positions differ by less than this are treated as same layer.
layer_tolerance = 0.20  # Å

layers = []

for z in z_sorted:
    if not layers:
        layers.append([z])
    elif abs(z - np.mean(layers[-1])) < layer_tolerance:
        layers[-1].append(z)
    else:
        layers.append([z])

layer_centers = np.array([np.mean(layer) for layer in layers])

print(f"Detected number of layers: {len(layers)}")

for i, layer in enumerate(layers):
    print(
        f"Layer {i+1:2d}: "
        f"z = {np.mean(layer):8.4f} Å, "
        f"atoms = {len(layer)}"
    )


# =========================
# 4. LAYER SPACING
# =========================
print("\n" + "=" * 60)
print("4. INTERLAYER SPACING")
print("=" * 60)

if len(layer_centers) > 1:
    spacings = np.diff(layer_centers)

    for i, d in enumerate(spacings):
        print(f"Layer {i+1} -> {i+2}: {d:.4f} Å")

    print(f"\nMean layer spacing: {np.mean(spacings):.4f} Å")
    print(f"Std. deviation     : {np.std(spacings):.4f} Å")

    if PD_BULK_A is not None:
        expected_d111 = PD_BULK_A / np.sqrt(3)

        print(f"\nExpected FCC Pd(111) spacing:")
        print(f"a_bulk / sqrt(3) = {expected_d111:.4f} Å")

        relative_error = (
            abs(np.mean(spacings) - expected_d111)
            / expected_d111
            * 100
        )

        print(f"Difference from expected = {relative_error:.2f}%")

        if relative_error < 5:
            print("PASS: layer spacing is consistent with Pd(111).")
        else:
            print("WARNING: layer spacing differs noticeably from Pd(111).")


# =========================
# 5. ESTIMATE VACUUM
# =========================
print("\n" + "=" * 60)
print("5. VACUUM ESTIMATE")
print("=" * 60)

# This simple calculation assumes c is close to the Cartesian z direction.
z_min = z_cart.min()
z_max = z_cart.max()

slab_thickness = z_max - z_min
cell_height = slab.lattice.c
vacuum = cell_height - slab_thickness

print(f"Atomic slab thickness: {slab_thickness:.4f} Å")
print(f"Cell c length        : {cell_height:.4f} Å")
print(f"Estimated vacuum     : {vacuum:.4f} Å")

if vacuum >= 10:
    print("PASS: vacuum is at least ~10 Å.")
else:
    print("WARNING: vacuum may be too small.")


# =========================
# 6. IN-PLANE GEOMETRY
# =========================
print("\n" + "=" * 60)
print("6. IN-PLANE GEOMETRY")
print("=" * 60)

print(f"a = {slab.lattice.a:.4f} Å")
print(f"b = {slab.lattice.b:.4f} Å")
print(f"gamma(a,b) = {slab.lattice.gamma:.4f} degrees")

gamma = slab.lattice.gamma

if abs(gamma - 60) < 5 or abs(gamma - 120) < 5:
    print(
        "PASS: a-b angle is consistent with a common "
        "hexagonal/triangular (111) surface cell."
    )
else:
    print(
        "NOTE: gamma is not ~60/120 degrees. "
        "This does NOT automatically mean the slab is wrong; "
        "a transformed or rectangular Pd(111) supercell is possible."
    )


# =========================
# FINAL SUMMARY
# =========================
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    """
For a good unrelaxed Pd(111) slab, you generally want:

- Pd only
- sensible Pd-Pd nearest-neighbor distances
- several well-defined atomic layers
- approximately equal interlayer spacing
- Pd(111) layer spacing close to a_bulk/sqrt(3)
- sufficient vacuum
- triangular/hexagonal atomic arrangement in the a-b view

Important:
A POSCAR/slab.vasp file usually does NOT contain the original Miller
index metadata, so this script cannot prove purely from metadata that
the requested surface was (111). The geometric checks above are what
you use to verify that.
"""
)