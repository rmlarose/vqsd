"""VQSD.py

Contains a class for VQSD circuits utilizing Cirq.
"""

# =============================================================================
# imports
# =============================================================================

import cirq
import numpy as np

# =============================================================================
# VQSD class
# =============================================================================

class VQSD:
    # =========================================================================
    # init method
    # =========================================================================

    def __init__(self, num_qubits, measure_key='z'):
        """Initializes a VQSD circuit.

        input:
            num_qubits [type: int]
                the number of qubits in the VQSD circuit

            measure_key [type: str]
                used to easily access measurement results with cirq

        data:
            qubits [type: cirq.QubitId]
                qubits in the circuit

            state_prep_circ [type: cirq.Circuit]
                state preperation part of the VQSD circuit

            unitray_circ [type: cirq.Circuit]
                diagonalizing unitary part of the VQSD circuit

            dip_test_circ [type: cirq.Circuit]
                dip test part of the VQSD circuit

            purity [type: float, initialized to None]
                the purity of the state being diagonalized
                once the circuit is formed, purity can be computed using the
                method self.compute_purity
        """
        # set the number of qubits and get some qubits
        # TODO: add option for mixed/pure state
        # for pure, only need 2 * num_qubits
        self._num_qubits = num_qubits
        self._total_num_qubits = 4 * self._num_qubits
        self.qubits = [cirq.LineQubit(ii)
                        for ii in range(self._total_num_qubits)]

        # key for measurements and statistics
        self._measure_key = measure_key
        self._pdip_key = "p"

        # initialize the circuits into logical components
        self.state_prep_circ = cirq.Circuit()
        self.unitary_circ = cirq.Circuit()
        self.dip_test_circ = cirq.Circuit()

        # initialize the purity of the state
        self.purity = None

    # =========================================================================
    # getter methods
    # =========================================================================

    def get_num_qubits(self):
        """Returns the number of qubits in the circuit."""
        return self._num_qubits

    # =========================================================================
    # methods to clear/reset circuits
    # =========================================================================
    
    def clear_state_prep_circ(self):
        """Sets the state prep circuit to be a new, empty circuit."""
        self.state_prep_circ = cirq.Circuit()
    
    def clear_unitary_circ(self):
        """Sets the unitary circuit to be a new, empty circuit."""
        self.unitary_circ = cirq.Circuit()
    
    def clear_dip_test_circ(self):
        """Sets the dip test circuit to be a new, empty circuit."""
        self.dip_test_circ = cirq.Circuit()

    # =========================================================================
    # circuit methods
    # =========================================================================

    def state_prep(self, prep_angles, post_angles, copy=0):
        """Adds the 'Arkin state prep' circuit to self.state_prep_circ.

        input:
            prep_angles [type: list<list<float>>]
                3 x self._num_qubits list of floats corresponding to the
                'half_turns' of the angles in the rotation gates.

                prep_angles[0] = x rotation half_angles
                prep_angles[1] = y rotation half_angles
                prep_angles[2] = z rotation half_angles

            post_angles
                    TODO: implement rotations after the CNOT layer
                    according to the post_angles
                    figure out the best structure for these

            copy [type: int, 0 or 1, default value = 0]
                the copy of the state rho to perform the state prep
                circuit on.

        modifies:
            self._state_prep_circ
        """
        # error check on inputs
        assert len(prep_angles) == 3
        assert len(prep_angles[0]) == self._num_qubits

        # =====================================================================
        # do the initial rotation layers
        # =====================================================================

        def rot_layer(rtype, nqubits, angles, copy=0):
            """Returns a rotation layer of type 'rtype' on 'nqubits' qubits
            with angles 'angles'.

            input:
                rtype ('rotation type') [type: str]
                    string key 'x' for R_x
                    string key 'y' for R_y
                    string key 'z' for R_z

                nqubits [type: int]
                    number of qubits in layer

                angles [type: list<floats>]
                    half_turns for each angle in layer
                    number of angles must be equal to the number of qubits

                copy [type: int (0 or 1, default value = 0)]
                    copy of the state to act on
            """
            # make sure we have the right number of angles
            assert len(angles) == nqubits

            # get the type of rotation gate
            if rtype.lower() == 'x':
                gate = cirq.RotXGate
            elif rtype.lower() == 'y':
                gate = cirq.RotYGate
            elif rtype.lower() == 'z':
                gate = cirq.RotZGate
            else:
                raise ValueError(
                    "unsupported rotation type. please enter x, y, or z"
                    )

            # get the layer
            for ii in range(nqubits):
                rot = gate(half_turns=angles[ii])
                yield rot(self.qubits[2 * nqubits * copy + ii])

        # append the rotation layers
        keylist = ['x', 'y', 'z']
        for (index, key) in enumerate(keylist):
            self.state_prep_circ.append(
                rot_layer(key,
                          self._num_qubits,
                          prep_angles[index],
                          copy),
                strategy=cirq.InsertStrategy.EARLIEST
                )

        # =====================================================================
        # do the cnot gates
        # =====================================================================

        for ii in range(self._num_qubits):
            ii += 2 * self._num_qubits * copy
            self.state_prep_circ.append(
                cirq.CNOT(self.qubits[ii],
                          self.qubits[ii + self._num_qubits]),
                strategy=cirq.InsertStrategy.EARLIEST
                )

        # =====================================================================
        # do the global rotations
        # =====================================================================

        # not sure if this is possible in cirq
        # TODO: figure this out
        # if global rotations are not possible, implement some 'standard form'
        # ask Arkin about this

    def product_state_prep(self, angles, rot_gate):
        """Adds a state preparation circuit of single qubit rotations.
        
        Args:
            angles : iterable
                Angles for the single qubit rotations.
                Number of angles must be equal to the number of qubits in
                the state.
            
            rot_gate : cirq.ops.Operation
                Single qubit rotation gate (RotXGate, RotYGate, etc.).
    
        Modifies: self.state_prep_circ
        """
        if len(angles) != self._num_qubits:
            raise ValueError("Incorrect number of angles.")
        
        n = self._num_qubits

        for ii in range(len(angles)):
            gate = rot_gate(half_turns=angles[ii])
            self.state_prep_circ.append(
                [gate(self.qubits[ii]), gate(self.qubits[ii + 2 * n])],
                strategy=cirq.InsertStrategy.EARLIEST
            )
    
    # =========================================================================
    # ansatz methods
    # =========================================================================

    def product_ansatz(self, params, gate):
        """Modifies self.unitary_circ by appending a product ansatz with a
        gate on each qubit.
        """
        # make sure the number of parameters is correct
        if len(params) != self._num_qubits:
            raise ValueError("Incorrect number of parameters.")
    
        n = self._num_qubits
        
        for ii in range(len(params)):
            g = gate(half_turns=params[ii])
            self.unitary_circ.append(
                [g(self.qubits[ii]), g(self.qubits[ii + 2 * n])],
                strategy=cirq.InsertStrategy.EARLIEST
            )
    
    def unitary(self, num_layers, params, shifted_params, copy):
        """Adds the diagonalizing unitary to self.unitary_circ.

        input:
            num_layers [type: int]
                number of layers to implement in the diagonalizing unitary.

            params [type: list<list<list<float>>>]
                parameters for every layer of gates
                the format of params is as follows:

                params = [[rotations for first layer],
                          [rotations for second layer],
                          ...,
                          [rotations for last layer]]

            shifted_params [type: list<list<list<float>>>]
                parameters for the shifted layers of gates
                format is the same as the format for params above

            copy [type: int, 0 or 1, default value = 0]
                the copy of the state to perform the unitary on
        """
        # TODO: implement
        pass

    def layer(self, params, shifted_params, copy):
        """Implements a single layer of the diagonalizing unitary.

        input:
            params [type: list<list<float>>]
                parameters for the first layer of gates.
                len(params) must be n // 2 where n is the number of qubits
                in the state and // indicates floor division.

                the format of params is as follows:

                params = [rotations for gates in layer]

                where the rotations for the gates in the layer have the form

                rotations for gates in layer =
                    [x1, y1, z1],
                    [x2, y2, z2],
                    [x3, y3, z3],
                    [x4, y4, z4].

                Note that each gate consists of 12 parameters. 3 parameters
                for each rotation and 4 total rotations.

                The general form for a gate, which acts on two qubits,
                is shown below:

                    ----------------------------------------------------------
                    | --Rx(x1)--Ry(y1)--Rz(z1)--@--Rx(x3)--Ry(y3)--Rz(z3)--@ |
                G = |                           |                          | |
                    | --Rx(x2)--Ry(y2)--Rz(z2)--X--Rx(x4)--Ry(y4)--Rz(z4)--X |
                    ----------------------------------------------------------

            shifted_params [type: ]
                TODO: figure this out
                parameters for the second shifted layer of gates

            copy [type: int (0 or 1)]
                the copy of the state to apply the layer to

        modifies:
            self.unitary_circ
                appends the layer of operations to self.unitary_circ
        """
        # for brevity
        n = self._num_qubits

        if params.size != self.num_angles_required_for_layer():
            raise ValueError("incorrect number of parameters for layer")

        # =====================================================================
        # helper functions for layer
        # =====================================================================

        def gate(qubits, params):
            """Helper function to append the two qubit gate
            ("G" in the VQSD paper figure).

            input:
                qubits [type: list<Qubits>]
                    qubits to be acted on. must have length 2.

                params [type: list<list<angles>>]
                    the parameters of the rotations in the gate.
                    len(params) must be equal to 12: 4 arbitrary rotations x
                    3 angles per arbitrary rotation.

                    the format of params must be

                    [[x1, y1, z1],
                     [x2, y2, z2],
                     [x3, y3, z3],
                     [x4, y4, z4]].

                    the general form of a gate, which acts on two qubits,
                    is shown below:

                    ----------------------------------------------------------
                    | --Rx(x1)--Ry(y1)--Rz(z1)--@--Rx(x3)--Ry(y3)--Rz(z3)--@ |
                G = |                           |                          | |
                    | --Rx(x2)--Ry(y2)--Rz(z2)--X--Rx(x4)--Ry(y4)--Rz(z4)--X |
                    ----------------------------------------------------------

            modifies:
                self.unitary_circ
                    appends a gate acting on the qubits to the unitary circ.
            """
            # rotation on 'top' qubit
            self.unitary_circ.append(
                self._rot(qubits[0], params[0]),
                strategy=cirq.InsertStrategy.EARLIEST
                )

            # rotation on 'bottom' qubit
            self.unitary_circ.append(
                self._rot(qubits[1], params[1]),
                strategy=cirq.InsertStrategy.EARLIEST
                )

            # cnot from 'top' to 'bottom' qubit
            self.unitary_circ.append(
                cirq.CNOT(qubits[0], qubits[1]),
                strategy=cirq.InsertStrategy.EARLIEST
                )

            # second rotation on 'top' qubit
            self.unitary_circ.append(
                self._rot(qubits[0], params[2]),
                strategy=cirq.InsertStrategy.EARLIEST
                )

            # second rotation on 'bottom' qubit
            self.unitary_circ.append(
                self._rot(qubits[1], params[3]),
                strategy=cirq.InsertStrategy.EARLIEST
                )

            # second cnot from 'top' to 'bottom' qubit
            self.unitary_circ.append(
                cirq.CNOT(qubits[0], qubits[1]),
                strategy=cirq.InsertStrategy.EARLIEST
                )

        # helper function for indexing loops
        stop = lambda n: n - 1 if n % 2 == 1 else n

        # shift in qubit indexing for different copies
        shift = 2 * self._num_qubits * copy

        # =====================================================================
        # implement the layer
        # =====================================================================

        # TODO: speedup. combine two loops into one

        # unshifted gates on adjacent qubit pairs
        for ii in range(0, stop(n), 2):
            iiq = ii + shift
            gate(self.qubits[iiq : iiq + 2], params[ii // 2])

        # shifted gates on adjacent qubits
        if n > 2:
            for ii in range(1, n, 2):
                iiq = ii + shift
                gate([self.qubits[iiq],
                      self.qubits[(iiq + 1) % n + shift]],
                     shifted_params[ii // 2])

    def _rot(self, qubit, params):
        """Helper function that returns an arbitrary rotation of the form
        R = Rz(params[2]) * Ry(params[1]) * Rx(params[0])
        on the qubit, e.g. R |qubit>.

        Note that order is reversed when put into the circuit. The circuit is:
        |qubit>---Rx(params[0])---Ry(params[1])---Rz(params[2])---
        """
        rx = cirq.RotXGate(half_turns=params[0])
        ry = cirq.RotYGate(half_turns=params[1])
        rz = cirq.RotZGate(half_turns=params[2])

        yield (rx(qubit), ry(qubit), rz(qubit))

    def dip_test(self, pdip=False):
        """Implements the dip test circuit.

        modifies:
            self.dip_test_circ
                appends the dip test circuit with measurements
                on the top state.
        """
        # TODO: implement option for partial dip test circuit
        # or make another method (e.g., pdip_test(self, qbit_to_measure))

        # do the cnots
        for ii in range(self._num_qubits):
            self.dip_test_circ.append(
                cirq.CNOT(self.qubits[ii + 2 * self._num_qubits],
                          self.qubits[ii]),
                strategy=cirq.InsertStrategy.EARLIEST
                )

        # do the measurements
        qubits_to_measure = self.qubits[:self._num_qubits]
        self.dip_test_circ.append(
            cirq.measure(*qubits_to_measure, key=self._measure_key),
            strategy=cirq.InsertStrategy.EARLIEST
        )
    
    def pdip_test(self, pdip_qubit_indices):
        """Implements the partial dip test circuit.
        
        Args:
            pdip_qubit_indices : list
                List of qubit indices (j in the paper) to do the pdip test on.
        
        Modifies:
            self.dip_test_circ
        """
        # do the cnots
        for ii in range(self._num_qubits):
            self.dip_test_circ.append(
                cirq.CNOT(self.qubits[ii + 2 * self._num_qubits],
                          self.qubits[ii]),
                strategy=cirq.InsertStrategy.EARLIEST
                )
        
        # add a Hadamard on each qubit not in the PDIP test
        all_qubit_indices = set(range(self._num_qubits))
        qubit_indices_to_hadamard = list(
            all_qubit_indices - set(pdip_qubit_indices)
        )
        qubits_to_hadamard = [self.qubits[ii + 2 * self._num_qubits]
                              for ii in qubit_indices_to_hadamard]
        self.dip_test_circ.append(
            cirq.H.on_each(qubits_to_hadamard)
        )
        
        # add the measurements for the dip test
        qubits_to_measure = [self.qubits[ii] for ii in pdip_qubit_indices]
        self.dip_test_circ.append(
            cirq.measure(*qubits_to_measure, key=self._measure_key),
            strategy=cirq.InsertStrategy.EARLIEST
        )
        
        # add the measurements for the destructive swap test on the pdip qubits
        pdip_qubits = [self.qubits[ii] for ii in qubit_indices_to_hadamard] \
                      + qubits_to_hadamard
        # edge case: no qubits in pdip set
        if len(pdip_qubits) > 0:
            self.dip_test_circ.append(
                cirq.measure(*pdip_qubits, key=self._pdip_key),
                strategy=cirq.InsertStrategy.EARLIEST
            )

    def state_overlap(self):
        """Returns a the state overlap circuit as a cirq.Circuit."""
        # declare a circuit
        circuit = cirq.Circuit()

        # gates to perform
        bell_basis_gates = lambda index: [
            cirq.CNOT(self.qubits[ii], 
                self.qubits[ii + 2 * self._num_qubits]),
                cirq.H(self.qubits[ii])
            ]

        # add the bell basis gates to the circuit
        for ii in range(self._num_qubits):
            circuit.append(
                bell_basis_gates(ii),
                strategy=cirq.InsertStrategy.EARLIEST
                )

        # measurements
        qubits_to_measure = self.qubits[ : self._num_qubits] + \
            self.qubits[2 * self._num_qubits : 3 * self._num_qubits]
        circuit.append(
            cirq.measure(*qubits_to_measure, key=self._measure_key)
            )

        return circuit

    # =========================================================================
    # helper circuit methods
    # =========================================================================

    def _get_unitary_symbols(self):
        """Returns a list of symbols required for the unitary ansatz."""
        # TODO: take into account the number of layers in the unitary
        # this should change how num_angles_required_for_unitary() is called
        # and the implementation of this method should change
        # the input arguments to this method should also include the number
        # of layers, as should num_angles_required_for_unitary()
        num_symbols_required = self.num_angles_required_for_unitary()
        return np.array(
            [cirq.Symbol(ii) for ii in range(num_symbols_required)]
            )

    def _reshape_sym_list_for_unitary(self):
        """Reshapes a one-dimensional list into the shape required by
        VQSD.layer.
        """
        pass

    def num_angles_required_for_unitary(self):
        """Returns the number of angles needed in the diagonalizing unitary."""
        # TODO: take into account the number of layers.
        # probably need to add a member variable to the class keeping track of
        # the number of layers.
        # should be 12 * num_qubits * num_layers
        return 12 * (self._num_qubits // 2)

    def num_angles_required_for_layer(self):
        """Returns the number of angles need in a single layer of the
        diagonalizing unitary.
        """
        return 12 * (self._num_qubits // 2)

    def state_overlap_postprocessing(self, output):
        """Does the classical post-processing for the state overlap algorithm.
        
        Args:
            output [type: np.array]
                The output of the state overlap algorithm.
                
                The format of output should be as follows:
                    vals.size = (number of circuit repetitions, 
                                 number of qubits being measured)

                    the ith column of vals is all the measurements on the
                    ith qubit. The length of this column is the number
                    of times the circuit has been run.
                    
        Returns:
            Estimate of the state overlap as a float
        """
        # =====================================================================
        # constants and error checking
        # =====================================================================

        # number of qubits and number of repetitions of the circuit
        (nreps, nqubits) = output.shape

        # check that the number of qubits is even
        assert nqubits % 2 == 0, "Input is not a valid shape."

        # initialize variable to hold the state overlap estimate
        overlap = 0.0

        # =====================================================================
        # postprocessing
        # =====================================================================

        # loop over all the bitstrings produced by running the circuit
        shift = nqubits // 2
        for z in output:
            parity = 1
            pairs = [z[ii] and z[ii + shift] for ii in range(shift)]
            # DEBUG
            for pair in pairs:
                parity *= (-1)**pair
            overlap += parity
            #overlap += (-1)**(all(pairs))

        return overlap / nreps

    # =========================================================================
    # methods for running the circuit and getting the objective function
    # =========================================================================

    def algorithm(self):
        """Returns the total algorithm of the VQSD circuit, which consists of
        state preperation, diagonalizing unitary, and dip test.

        rtype: cirq.Circuit
        """
        return self.state_prep_circ + self.unitary_circ + self.dip_test_circ

    def resolved_algorithm(self, angles):
        """Returns the total algorithm of the VQSD circuit with all
        parameters resolved.

        Args:
            angles [type: array like]
                list of angles in the diagonalizing unitary.
        """
        circuit = self.algorithm()

        if angles is None:
            angles = 2 * np.random.rand(12 * self._num_qubits)
        param_resolver = cirq.ParamResolver(
            {str(ii) : angles[ii] for ii in range(len(angles))}
        )

        return circuit.with_parameters_resolved_by(param_resolver)

    def run(self,
            simulator=cirq.google.XmonSimulator(),
            repetitions=1000):
        """Runs the algorithm and returns the result.

        rtype: cirq.TrialResult
        """
        return simulator.run(self.algorithm(), repetitions=repetitions)

    def run_resolved(self,
                     angles,
                     simulator=cirq.google.XmonSimulator(),
                     repetitions=1000):
        """Runs the resolved algorithm and returns the result."""
        return simulator.run(
            self.resolved_algorithm(angles), repetitions=repetitions
        )

    def obj_dip(self,
                simulator=cirq.google.XmonSimulator(),
                repetitions=1000):
        """Returns the objective function as computed by the DIP Test."""
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()

        # run the circuit
        outcome = self.run(simulator, repetitions)
        counts = outcome.histogram(key=self._measure_key)
        
        # compute the overlap and return the objective function
        overlap = counts[0] / repetitions if 0 in counts.keys() else 0
        return self.purity - overlap

    def obj_dip_resolved(self,
                         angles,
                         simulator=cirq.google.XmonSimulator(),
                         repetitions=1000):
        """Returns the objective function of the resolved circuit as computed
        by the DIP Test."""
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()

        # run the circuit
        outcome = self.run_resolved(angles, simulator, repetitions)
        counts = outcome.histogram(key=self._measure_key)

        # compute the overlap and return the objective
        overlap = counts[0] / repetitions if 0 in counts.keys() else 0
        return self.purity - overlap
    
    def overlap_pdip(self,
                     simulator=cirq.google.XmonSimulator(),
                     repetitions=1000):
        """Returns the objective function as computed by the PDIP Test."""
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()
        
        # store the overlap
        ov = 0.0
        
        for j in range(self._num_qubits):
            # do the appropriate pdip test circuit
            self.clear_dip_test_circ()
            self.pdip_test([j])
            
            # DEBUG
            print("j =", j)
            print("PDIP Test Circuit:")
            print(self.dip_test_circ)
            
            # run the circuit
            outcome = self.run(simulator, repetitions)
            
            # get the measurement counts
            dipcounts = outcome.measurements[self._measure_key]
            pdipcount = outcome.measurements[self._pdip_key]
            
            # postselect on the all zeros outcome for the dip test measuremnt
            mask = self._get_mask_for_all_zero_outcome(dipcounts)
            toprocess = pdipcount[mask]
            
            # do the state overlap (destructive swap test) postprocessing
            overlap = self.state_overlap_postprocessing(toprocess)
            
            # DEBUG
            print("Overlap = ", overlap)
            
            # divide by the probability of getting the all zero outcome
            prob = len(np.where(mask == True)) / len(mask)
            counts = outcome.histogram(key=self._measure_key)
            prob = counts[0] / repetitions if 0 in counts.keys() else 0.0
            
            assert 0 <= prob <= 1
            print("prob =", prob)
            
            
            overlap *= prob
            print("Scaled overlap =", overlap)
            print()
            ov += overlap

        return ov / self._num_qubits
    
    def overlap_pdip_resolved(self,
                              angles,
                              simulator=cirq.google.XmonSimulator(),
                              repetitions=1000):
        """Returns the objective function as computed by the PDIP Test
        for the input angles in the ansatz."""
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()
        
        # store the overlap
        ov = 0.0
        
        for j in range(self._num_qubits):
            # do the appropriate pdip test circuit
            self.clear_dip_test_circ()
            self.pdip_test([j])
            
            # DEBUG
            #print("j =", j)
            #print("PDIP Test Circuit:")
            #print(self.dip_test_circ)
            
            # run the circuit
            outcome = self.run_resolved(angles, simulator, repetitions)
            
            # get the measurement counts
            dipcounts = outcome.measurements[self._measure_key]
            pdipcount = outcome.measurements[self._pdip_key]
            
            # postselect on the all zeros outcome for the dip test measuremnt
            mask = self._get_mask_for_all_zero_outcome(dipcounts)
            toprocess = pdipcount[mask]
            
            # do the state overlap (destructive swap test) postprocessing
            overlap = self.state_overlap_postprocessing(toprocess)
            
            # DEBUG
            #print("Overlap = ", overlap)
            
            # divide by the probability of getting the all zero outcome
            counts = outcome.histogram(key=self._measure_key)
            prob = counts[0] / repetitions if 0 in counts.keys() else 0.0
            
            assert 0 <= prob <= 1
            #print("prob =", prob)
            
            
            overlap *= prob
            #print("Scaled overlap =", overlap)
            #print()
            ov += overlap
            
        return ov / self._num_qubits

    def obj_pdip(self,
                 simulator=cirq.google.XmonSimulator(),
                 repetitions=1000):
        """Returns the purity of the state - the overlap as computed by the
        PDIP Test.
        """
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()

        return self.purity - self.overlap_pdip(simulator, repetitions)
    
    def obj_pdip_resolved(self,
                          angles,
                          simulator=cirq.google.XmonSimulator(),
                          repetitions=1000):
        """Returns the purity of the state - the overlap as computed by the
        PDIP Test for the input angles in the ansatz.
        """
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()

        return self.purity - self.overlap_pdip_resolved(angles, simulator, repetitions)
            
    def _get_mask_for_all_zero_outcome(self, outcome):
        """Returns a mask corresponding to indices ii from 0 to len(outcome) 
        such that np.all(outcome[ii]) == True.
        
        Args:
            outcome : numpy.ndarray 
                The output of the state overlap algorithm.
                
                The format of output should be as follows:
                    outcome.size = (number of circuit repetitions, 
                                 number of qubits being measured)

                    the ith column of outcome is all the measurements on the
                    ith qubit. The length of this column is the number
                    of times the circuit has been run.
        """
        mask = []
        for meas in outcome:
            if not np.any(meas):
                mask.append(True)
            else:
                mask.append(False)
        return np.array(mask)

    def compute_purity(self,
                       simulator=cirq.google.XmonSimulator(),
                       repetitions=10000):
        """Computes and returns the (approximate) purity of the state."""
        # get the circuit without the diagonalizing unitary
        circuit = self.state_prep_circ + self.state_overlap()
        # DEBUG
        print("I'm computing the purity as per the circuit:")
        print(circuit)
        outcome = simulator.run(circuit, repetitions=repetitions)
        vals = outcome.measurements[self._measure_key]
        self.purity = self.state_overlap_postprocessing(vals)

    def init_state_to_matrix(self):
        """Returns the initial state defined by the state preperation
        circuit in matrix form. This corresponds to \rho in the notation
        of the VQSD paper.
        """
        # TODO: implement

    def diag_state_to_matrix(self):
        """Returns the state in matrix form after the diagonalizing unitary
        has been applied to the input state. This corresponds to \rho' in the
        notation of the VQSD paper.
        """
        # TODO: implement

    # =========================================================================
    # overrides
    # =========================================================================

    def __str__(self):
        """Returns the VQSD circuit's algorithm."""
        return self.algorithm().to_text_diagram()


def min_to_vqsd(param_list, num_qubits=2):
    """Helper function that converts a linear array of angles (used to call
    the optimize function) into the format required by VQSD.layer.
    """
    # TODO: add this as a member function of VQSD class
    # error check on input
    assert len(param_list) % 6 == 0, "invalid number of parameters"
    return param_list.reshape(num_qubits // 2, 4, 3)

def vqsd_to_min(param_array):
    """Helper function that converts the array of angles in the format
    required by VQSD.layer into a linear array of angles (used to call the
    optimize function).
    """
    # TODO: add this as a member function of VQSD class
    return param_array.flatten()

def symbol_list(num_qubits, num_layers):
    """Returns a list of cirq.Symbol's for the diagonalizing unitary."""
    return np.array(
        [cirq.Symbol(str(ii)) for ii in range(12 * (num_qubits // 2) * num_layers)]
        )

def symbol_list_for_product(num_qubits):
    """Returns a list of cirq.Symbol's for a product state ansatz."""
    return np.array(
        [cirq.Symbol(str(ii)) for ii in range(num_qubits)]
    )

def get_param_resolver(num_qubits, num_layers):
    """Returns a cirq.ParamResolver for the parameterized unitary."""
    num_angles = 12 * num_qubits * num_layers
    angs = np.pi * (2 * np.random.rand(num_angles) - 1)
    return cirq.ParamResolver(
        {str(ii) : angs[ii] for ii in range(num_angles)}
    )