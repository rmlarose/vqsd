"""testVQSD.py

Test cases for VQSD class.
"""

# =============================================================================
# imports
# =============================================================================

import time
import cirq
import numpy as np

from VQSD import VQSD, min_to_vqsd, vqsd_to_min, symbol_list

# =============================================================================
# constants
# =============================================================================

SEP_STRING = "="
CHAR_LEN = 80
DOT_LEN = 50
DOT = "."

# =============================================================================
# helper functions
# =============================================================================

def print_sep(title, char_len=CHAR_LEN, sep_string=SEP_STRING):
    """Prints separation strings."""
    print("".center(char_len, sep_string))
    print(title)
    print("".center(char_len, sep_string))

# =============================================================================
# unit tests
# =============================================================================

def test_num_qubits(num_tests=100, maxn=30):
    """Tests the getter method VQSD.get_num_qubits(). If this is failing
    something's gone horribly wrong.
    """
    for _ in range(num_tests):
        n = np.random.randint(1, maxn)
        circ = VQSD(n)
        assert circ.get_num_qubits() == n

    print("\ntest_num_qubits()".ljust(DOT_LEN, DOT), "passed!", sep="")

def test_angle_format_conversions(maxn=5):
    """Tests min_to_vqsd and vqsd_to_min functions in runVQSD."""
    for n in range(2, maxn + 1):
        params = np.array((n // 2)
                          * [[[.1, .2, .3], [1, .2, .3],
                              [.3, .2, 1], [.4, .2, .1]]]
                         )
        min_params = vqsd_to_min(params)

        assert (vqsd_to_min(min_to_vqsd(min_params, n)) == min_params).all()

    print("test_angle_format_conversions()".ljust(DOT_LEN, DOT),
          "passed!",
          sep="")

def test_number_of_params(nqubits=2, nlayers=1):
    """Tests that the number of parameters is correct for a diagonalizing
    unitary with a given number of qubits and layers.
    
    Args:
        nqubits [type: int]
            number of qubits in state
        
        nlayers [type: int]
            number of layers in diagonalizing unitary
    """
    pass

def test_two_qubit_state_identity(repetitions=1000, verbose=False):
    """Tests the minimum distance is acheived for a circuit that
    does nothing."""
    # number of qubits and layers in the unitary
    n = 2
    num_layers = 1

    # state preperation angles
    prep_angles = [[0, 0], [0, 0], [0, 0]]
    post_angles = []

    # symbols for a single layer in the diagonalizing unitary
    params = min_to_vqsd(symbol_list(n, num_layers))
    shifted_params = params

    # =========================================================================
    # get a VQSD circ
    # =========================================================================

    # make a VQSD circuit on n qubits
    circ = VQSD(n)

    # add the state preperation
    circ.state_prep(prep_angles, post_angles, copy=0)
    circ.state_prep(prep_angles, post_angles, copy=1)

    # add the unitary
    for _ in range(num_layers):
        circ.layer(params, shifted_params, copy=0)
        circ.layer(params, shifted_params, copy=1)

    # add the dip test
    circ.dip_test()

    # print the algorithm with symbols
    if verbose:
        print("Structure of Circuit:\n", circ.algorithm())
        print("\n")

    # print the algoirthm with resolved angles
    test_angles = [0.0] * circ.num_angles_required_for_unitary()
    if verbose:
        print("Circuit with instantiated parameters:\n",
              circ.resolved_algorithm(test_angles))
        print("\n")

    # get the objective and time how long it takes
    start = time.time()
    distance = circ.obj_dip_resolved(test_angles, repetitions=repetitions)
    runtime = time.time() - start

    # print the objective and the timing info
    if verbose:
        print("Now running the circuit with instantiated parameters...")
        print("Total runtime = {} seconds".format(runtime))
        print("HS distance between the two states = ", distance)

    # make sure the overlap is close to one
    tolerance = 1.0 / repetitions
    assert distance < tolerance

    # print success message
    print("test_two_qubit_state_identity()".ljust(DOT_LEN, DOT),
          "passed!",
          sep="")

def test_two_qubit_product_state_identity(
        half_turn=0.1, repetitions=1000,
        verbose=False):
    """Tests the VQSD algorithm for two qubit pure product states.
    State preperation is single qubit rotations.
    Diagonalizing unitary is the inverse of state prep.

    Tests to see if overlap is maximal.
    """
    # get a VQSD circuit
    circ = VQSD(2)

    # define the rotations
    rot = cirq.RotXGate(half_turns=half_turn)
    rotdag = cirq.RotXGate(half_turns=-half_turn)

    # add the state preperation
    circ.state_prep_circ.append(
        [rot(circ.qubits[x]) for x in [0, 1, 4, 5]]
        )

    # add the unitary
    circ.unitary_circ.append(
        [rotdag(circ.qubits[x]) for x in [0, 1, 4, 5]]
        )

    # add the dip test circuit
    circ.dip_test()

    # verbose output
    if verbose:
        print("The total circuit is", circ.algorithm(), sep="\n")

    # get the HS distance
    distance = circ.obj_dip(repetitions=repetitions)

    # make sure we're close
    tolerance = 1 / repetitions
    assert distance < tolerance

    # print success message
    print("test_two_qubit_product_state_identity()".ljust(DOT_LEN, DOT),
          "passed!",
          sep="")

def test_n_qubit_product_state_identity(
        n=3, rot_type='x',
        half_turn=0.1, verbose=False):
    """Tests the VQSD algorithm for n qubit pure product states."""
    # TODO: implement!
    pass

def test_dip_test_n_qubits_identity(n=2, repetitions=1000, verbose=False):
    """Runs the DIP Test only and tests overlap is maximal."""
    circ = VQSD(n)
    circ.dip_test()
    if verbose:
        print(circ.algorithm())
    obj = circ.obj_dip(repetitions=repetitions)
    if verbose:
        print("objective = ", obj)
    tolerance = 1 / repetitions
    assert obj < tolerance

def test_dip_test_identity_loop(maxn=10, repetitions=1000, verbose=False):
    """Loops over numbers of qubits and tests that the DIP Test acheives
    maximal overlap for a circuit with only the DIP Test circuit.
    """
    # loop through circuit sizes and test the dip test
    for n in range(2, maxn):
        test_dip_test_n_qubits_identity(
            n=n, repetitions=repetitions, verbose=verbose
            )

    # print success message
    print("test_dip_test_identity_loop()".ljust(DOT_LEN, DOT),
          "passed!",
          sep="")

def test_purity_pure_state(repetitions=10000, verbose=False):
    """Forms a VQSD circuit diagonalizing a pure state and asserts the purity
    is one.
    """
    # number of qubits
    n = 2

    # state preperation angles
    prep_angles = [[0, 0], [0, 0], [0, 0]]
    post_angles = []

    # make a VQSD circuit on n qubits
    circ = VQSD(n)

    # add the state preperation
    circ.state_prep(prep_angles, post_angles, copy=0)
    circ.state_prep(prep_angles, post_angles, copy=1)

    # add the dip test
    circ.dip_test()

    # print the algorithm with symbols
    if verbose:
        print("Structure of Circuit:\n", circ.algorithm())
        print("\n")

    # compute the purity of the state
    circ.compute_purity(repetitions=10000)

    # print the purity
    if verbose:
        print("the purity of the state is:", circ.purity)

    tolerance = 1 / repetitions
    assert abs(circ.purity - 1) < tolerance

    # print success message
    print("test_purity_pure_state()".ljust(DOT_LEN, DOT), "passed!", sep="")

def test_state_overlap_visual(num_qubits=4):
    """Computes the state overlap circuit and prints it to the console for
    visual verification of correctness.
    """
    print(VQSD(num_qubits).state_overlap())

def test_state_overlap_basic(n=2, nreps=1000, random=False, verbose=False):
    """Basic test for state overlap.
    
    Checks that the purity of a pure state is one and the purity of a
    mixed state is less than one.
    """
    # state preperation angles
    if random:
        prep_angles = [np.random.rand(n)] * 3
    else:
        prep_angles = [np.zeros(n)] * 3
    post_angles = []

    # make a VQSD circuit on n qubits
    circ = VQSD(n)

    # add the state preperation
    circ.state_prep(prep_angles, post_angles, copy=0)
    circ.state_prep(prep_angles, post_angles, copy=1)

    # get rid of the CNOTS in the state preperation to test pure states
    circ.state_prep_circ = circ.state_prep_circ[: -2]

    # add the state overlap circuit
    circ.dip_test_circ = circ.state_overlap()

    # print the circuit if verbose output is desired
    if verbose:
        print("The algorithm is:", circ.algorithm(), sep="\n\n")

    # get the circuit output bit strings
    out = circ.run(repetitions=nreps)
    vals = out.measurements[circ._measure_key]

    # do the postprocessing
    purity = circ.state_overlap_postprocessing(vals)
    if verbose:
        print("purity = ", purity)

    # check that the purity is unity
    assert abs(purity - 1) < 1 / nreps

def test_state_overlap_basic_loop(maxn=8, nreps=1000,
    random=False, verbose=False):
    """Runs test_state_overlap_basic for many circuit sizes."""
    for n in range(2, maxn):
        test_state_overlap_basic(n=n, nreps=nreps,
            random=random, verbose=verbose)

    # print success message
    print("test_state_overlap_basic_loop()".ljust(DOT_LEN, DOT),
        "passed!", sep="")

def test_purity_analytic(verbose=False):
    """Tests that the computed purity for a state whose analytic purity
    we know is correct.
    """
    # number of qubits
    n = 2

    # prep angles in circuit
    pangles = np.array([
        [0, 0],
        [0.5, 0.5],
        [0, 0]
        ])

    # make a VQSD circuit on n qubits
    circ = VQSD(n)

    # add the state preperation
    circ.state_prep(pangles, None, copy=0)
    circ.state_prep(pangles, None, copy=1)

    # verbose output
    if verbose:
        # print the algorithm with symbols
        print("Structure of Circuit:\n", circ.algorithm())
        print("\n")

    # compute the purity of the state
    circ.compute_purity(repetitions=100000)
    print("the purity of the state is:", circ.purity)

    # make sure it's close to the actual value

def test_pdip_visual(indices, verbose=True):
    """Visual test for PDIP Test circuit."""
    circ = VQSD(4)
    
    circ.pdip_test(indices)

    if verbose:
        print("Circuit for PDIP Test with indices", indices)
        print(circ.dip_test_circ)
    
    # print success message
    print("test_pdip_visual()".ljust(DOT_LEN, DOT),
          "passed!", sep="")

def test_get_mask_for_all_zero_outcome(n=4, verbose=False):
    """Tests that VQSD._get_mask_for_all_zero_outcome(...) returns the correct
    mask.
    """
    circ = VQSD(n)
    
    # do the pdip test with the zeroth qubit
    circ.pdip_test([0])
    
    # verbose option
    if verbose:
        print(circ.algorithm())
    
    # simulate and measure
    sim = cirq.google.XmonSimulator()
    res = sim.run(circ.algorithm(), repetitions=100)
    
    # split the measurements
    dipm = res.measurements["z"]
    pdipm = res.measurements["p"]
    
    assert not np.any(dipm)
    
    mask = circ._get_mask_for_all_zero_outcome(dipm)
    assert np.all(mask)
    
    assert np.all(pdipm[mask] == pdipm)
    
    # print success message
    print("test_get_mask_for_all_zero_outcome()".ljust(DOT_LEN, DOT),
          "passed!", sep="")
    
def test_get_mask_for_all_zero_outcome2(n=4, verbose=False):
    """Tests that VQSD._get_mask_for_all_zero_outcome(...) returns the correct
    mask.
    """
    circ = VQSD(n)
    
    # flip the zeroth qubit
    circ.state_prep_circ.append(cirq.H(circ.qubits[0]))
    
    # do the pdip test
    circ.pdip_test([0])
    
    # verbose option
    if verbose:
        print(circ.algorithm())
    
    # simulate and measure
    sim = cirq.google.XmonSimulator()
    res = sim.run(circ.algorithm(), repetitions=100)
    
    # split the measurements
    dipm = res.measurements["z"]
    pdipm = res.measurements["p"]
    
    mask = circ._get_mask_for_all_zero_outcome(dipm)
    assert len(mask) == len(dipm)
    
    numel = len(np.where(mask == True))
    
    assert len(pdipm[mask] == numel)
    assert len(pdipm[mask]) < len(pdipm)
    
    # print success message
    print("test_get_mask_for_all_zero_outcome2()".ljust(DOT_LEN, DOT),
          "passed!", sep="")

def test_overlap_pdip(n=2):
    """Tests that the overlap for a pure state
    as computed by the PDIP Test is 1."""
    circ = VQSD(n)
    assert np.isclose(circ.overlap_pdip(), 1.0)
    # print success message
    print("test_overlap_pdip()".ljust(DOT_LEN, DOT),
          "passed!", sep="")

def test_obj_pdip(n=2):
    """Tests that the cost for |0><0| as computed by the PDIP Test is 0."""
    circ = VQSD(n)
    circ.compute_purity()
    assert np.isclose(circ.obj_pdip(), 0.0)
    # print success message
    print("test_obj_pdip()".ljust(DOT_LEN, DOT),
          "passed!", sep="")

def test_overlap_pdip2(n=2):
    """Tests that the overlap for the |++> state."""
    circ = VQSD(n)
    circ.state_prep_circ.append(
        cirq.H.on_each(circ.qubits[:n] + circ.qubits[2 * n : 3 * n])
    )
    circ.compute_purity()
    print("overlap =", circ.overlap_pdip())
    assert abs(circ.overlap_pdip() - 0.5) < 5e-2
    # print success message
    print("test_overlap_pdip()".ljust(DOT_LEN, DOT),
          "passed!", sep="")

def test_obj_pdip_diagonal():
    """Tests the objective as computed by the PDIP Test is zero for a
    diagonal state on four qubits."""
    circ = VQSD(4)
    circ.state_prep_circ.append(
        cirq.X.on_each(circ.qubits[:4] + circ.qubits[8:12])
    )
    print(circ.algorithm())
    circ.compute_purity()
    assert np.isclose(circ.obj_pdip(), 0.0)
    
    # print success message
    print("test_obj_pdip_diagonal()".ljust(DOT_LEN, DOT),
          "passed!", sep="")
    
def test_obj_pdip_diagonal2():
    """Tests the objective as computed by the PDIP Test is zero for a
    diagonal state on four qubits."""
    circ = VQSD(4)

    print(circ.algorithm())
    circ.compute_purity()
    assert np.isclose(circ.obj_pdip(), 0.0)
    
    # print success message
    print("test_obj_pdip_diagonal2()".ljust(DOT_LEN, DOT),
          "passed!", sep="")
# =============================================================================
# run the tests
# =============================================================================

if __name__ == "__main__":
    # TODO: grab input arguments for options (e.g., verbose, what tests to
    # run, etc.)

    print_sep("now testing VQSD")

    test_num_qubits()
    test_angle_format_conversions()
    test_two_qubit_state_identity(repetitions=100)
    test_two_qubit_product_state_identity()
    test_dip_test_identity_loop(maxn=8)
    test_purity_pure_state()
    test_state_overlap_basic_loop()
    test_pdip_visual([0], verbose=True)
    test_get_mask_for_all_zero_outcome()
    test_get_mask_for_all_zero_outcome2()
    test_overlap_pdip()
    test_obj_pdip()
    test_overlap_pdip2()
    test_obj_pdip_diagonal()
    test_obj_pdip_diagonal2()

    print("\n")
    print_sep("all tests for VQSD passed!")
