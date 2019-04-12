"""Tests the local cost vs global cost in VQSD."""

# =============================================================================
# Imports
# =============================================================================

import time

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import cirq

from VQSD import VQSD, symbol_list_for_product

# =============================================================================
# Constants
# =============================================================================

n = 10
nreps = 500
method="COBYLA"
q = 0.5
maxiter = 1000

# =============================================================================
# Functions
# =============================================================================

def process(vals):
    new = [vals[0]]
    for ii in range(1, len(vals)):
        if vals[ii] < new[-1]:
            new.append(vals[ii])
        else:
            new.append(new[-1])
    return new

# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":    
    # Arrays to store the cost values
    OBJDIPS = []        # global cost trained with local cost
    OBJPDIPS = []       # local cost trained with local cost
    OBJGLOBALDIPS = []  # global cost trained with global cost
    OBJQDIPS = []       # global cost trained with q cost
    QOBJS = []          # weighted sum of local and global cost

    # Get a VQSD instance
    vqsd = VQSD(n)
    
    #np.random.seed(8675310)
    
    # Get preparation angles
    sprep_angles = np.random.rand(n)
    
    #sprep_angles = [0.75] * n
    
    # Add the state prep circuit and compute the purity
    vqsd.product_state_prep(sprep_angles, cirq.RotXGate)
    vqsd.compute_purity()
    
    # Add the ansatz
    vqsd.product_ansatz(symbol_list_for_product(n), cirq.RotXGate)
    
    # Objective function for Dip Test
    def objdip(angs):
        vqsd.clear_dip_test_circ()
        vqsd.dip_test()
        val = vqsd.obj_dip_resolved(angs, repetitions=nreps)
        OBJGLOBALDIPS.append(val)
        print("DIP Test obj =", val)
        return val

    # Objective function for PDIP Test
    def objpdip(angs):
        val = vqsd.obj_pdip_resolved(angs, repetitions=nreps)
        OBJPDIPS.append(val)
        print("PDIP Test obj =", val)
        return val
    
    # Does the PDIP and also appends the DIP
    def objpdip_compare(angs):
        # Do the PDIP first
        pval = vqsd.obj_pdip_resolved(angs, repetitions=nreps)
        OBJPDIPS.append(pval)
        print("\nPDIP Test obj =", pval)
        
        # Do the DIP Test next. Evaluate with many repetitions to get a
        # good estimate of the cost here. We're not training with this,
        # just evaluating the cost
        vqsd.clear_dip_test_circ()
        vqsd.dip_test()
        val = vqsd.obj_dip_resolved(angs, repetitions=10000)
        OBJDIPS.append(val)
        print("DIP Test obj =", val)
        
        # return the PDIP Test obj val
        return pval
    
    # Does the weighted sum of costs
    def qcost(angs):
        # PDIP cost
        pdip = vqsd.obj_pdip_resolved(angs, repetitions=nreps)
        
        # DIP cost
        dip = vqsd.obj_dip_resolved(angs, repetitions=nreps)
        
        # weighted sum
        obj = q * dip + (1 - q) * pdip
        
        QOBJS.append(obj)
        print("QCOST OBJ =", obj)
        
        vqsd.clear_dip_test_circ()
        vqsd.dip_test()
        val = vqsd.obj_dip_resolved(angs, repetitions=10000)
        OBJQDIPS.append(val)
        
        return obj
    
    # =========================================================================
    # Do the optimization
    # =========================================================================
    
    # Initial values
    init = np.zeros(n)

    # Start the timer
    start = time.time()
    
    # Minimize using the local cost + evaluate the global cost at each iteration
    out = minimize(objpdip_compare, init, method=method, options={"maxiter": maxiter})
    
    # Minimize using the global cost
    glob = minimize(objdip, init, method=method, options={"maxiter": maxiter})
    
    # Minimize using the weighted cost
    weight = minimize(qcost, init, method=method, options={"maxiter": maxiter})
    
    print("PDIP angles:", [x % 2 for x in out["x"]])
    print("DIP angles:", [x % 2 for x in glob["x"]])
    print("Actual angles:", sprep_angles)
    
    # Print the runtime
    wall = (time.time() - start) / 60
    print("Runtime {} minutes".format(wall))
    
    # =========================================================================
    # Do the plotting
    # =========================================================================
    
    plt.figure(figsize=(6, 7))
    title = "EXACT GLOBAL EVAL {} {} Qubit Product State, {} Shots, {} Iterations, Runtime = {} min.".format(method, n, nreps, maxiter, round(wall, 2))
    plt.title(title)
    
    plt.plot(process(OBJPDIPS), "-o", linewidth=2, label="Local trained with local")
    plt.plot(process(OBJDIPS), "-o", linewidth=2, label="Global trained with local")
    plt.plot(process(OBJGLOBALDIPS), "-o", linewidth=2, label="Global trained with global")
    plt.plot(process(QOBJS), "-o", linewidth=2, label="q = {}".format(q))
    plt.plot(process(OBJQDIPS), "-o", linewidth=2, label="Global trained with q = {}".format(q))
    
    plt.grid()
    plt.legend()
    
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    
    # Save the figure
    t = time.asctime()
    plt.savefig(title + t + ".pdf", format="pdf")
    
    # =========================================================================
    # Write the data to a text file
    # =========================================================================

    costs = [process(OBJPDIPS),
             process(OBJDIPS),
             process(OBJGLOBALDIPS),
             process(QOBJS)]
    
    # Pad the lengths
    maxlen = max([len(a) for a in costs])
    for a in costs:
        while len(a) < maxlen:
            a.append(a[-1])

    data = np.array(costs)
    
    fname = title + t + ".txt"
    
    np.savetxt(fname, data.T)
