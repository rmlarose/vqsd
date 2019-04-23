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
pattern = "EXACT*4 Qubit Product State, 500 Shots*Mon*.txt"

# Tolerances to consider
tols = np.linspace(0.01, 0.2, 40)

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
plt.plot(tols, counts[:, 1], "--s",
         linewidth=2, markersize=4, markeredgewidth=2,
         label="Global trained with local")
plt.plot(tols, counts[:, 2], "--s",
         linewidth=2, markersize=4, markeredgewidth=2,
         label="Global trained with global")
plt.plot(tols, counts[:, 4], "--s",
         linewidth=2, markersize=4, markeredgewidth=2,
         label="Global trained with q=0.5")

plt.ylabel("Fraction of successful optimizations", fontsize=14, fontweight="bold")
plt.xlabel("Tolerance", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(which="both")
plt.show()
