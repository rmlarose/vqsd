# Introduction

Code repository for the Variational Quantum State Diagonalization algorithm as explained in [Variational Quantum State Diagonalization](https://arxiv.org/abs/1810.10506).

# Requirements

This code is written for Cirq v0.3. To install this version of Cirq, type the following at a command line (note: you may wish to be in a virtual environment):

`pip install cirq==0.3.0`

This should install *most* external dependencies needed. For some scripts, other dependencies (like matplotlib) may be needed.

# Files

The file `VQSD.py` contains the definition of the `VQSD` class. 

The file `testVQSD.py` contains unit tests for this class. You can also find examples of how to use the class in these unit tests.

The file `localVQSD.py` is a script which tests the local vs global cost performance (see paper for details). This is also an example of how to use the `VQSD` class.

