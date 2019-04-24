"""Count the number of successful optimizations for local vs global cost
in VQSD on a four qubit product state.
"""

# =============================================================================
# Imports
# =============================================================================

import glob
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Constants
# =============================================================================

# For finding the right file names
pattern = "EXACT*6 Qubit Product State, 500 Shots*.txt"

# Tolerances to consider
tols = np.linspace(0.001, 0.2, 50)

# Array to store number of successful optimizations
# Column 0 = local trained with local
# Column 1 = global trained with local
# Column 2 = global trained with global
# Column 3 = q = 0.5
# Column 4 = global trained with q = 0.5
counts = np.zeros((len(tols), 5))

# List of all filenames
fnames = glob.glob(pattern)
print("Number of runs =", len(fnames))

# =============================================================================
# Read in files and get statistics
# =============================================================================

# Loop over tolerances
for (ii, tol) in enumerate(tols):
    # Loop through all filenames
    for fname in fnames:
        # Load the data
        data = np.loadtxt(fname)

	# Only keep data with all five relevant costs
        if data.shape[1] != 5:
            continue
        
        # Get the minima in each column
        mins = np.min(data, axis=0)

        # Add the successful optimizations
        counts[ii] += mins <= tol
    
# Compute the final stats
counts /= len(fnames)

    
# =============================================================================
# Do the plotting
# =============================================================================

plt.figure(figsize=(6, 7))
"""
plt.plot(tols, counts[:, 0], "--s", 
         linewidth=2, markersize=4, markeredgewidth=2,
         label="Local trained with local")
"""
# Global trained with local
plt.plot(tols, counts[:, 1], "--s", color="chartreuse",
         linewidth=2, markersize=4, markeredgewidth=2,
         label="$C(q = 1.0)$ with $C(q = 0.0)$ training")

# Global trained with global
plt.plot(tols, counts[:, 2], "--s", color="green",
         linewidth=2, markersize=4, markeredgewidth=2,
         label="$C(q = 1.0)$ with $C(q = 1.0)$ training")

"""
plt.plot(tols, counts[:, 3], "--s",
         linewidth=2, markersize=4, markeredgewidth=2,
         label="$C(q = 1.0)$ with $C(q = 0.5)$ training")
"""

# Global trained with q = 0.5 cost
plt.plot(tols, counts[:, 4], "--s", color="purple",
         linewidth=2, markersize=4, markeredgewidth=2,
         label="$C(q = 1.0)$ with $C(q = 0.5)$ training")

# Axes, legend, and grid
plt.ylabel("Fraction of successful optimizations", fontsize=14, fontweight="bold")
plt.ylim((0, 1))
plt.xlabel("Tolerance", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(which="both")
plt.show()
