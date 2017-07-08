# -*- coding: utf-8 -*-
"""qudotpy.hamiltonians

Module to build common hamiltonians for study.

:copyright: Copyright (C) 2017 Perry Sakkaris <psakkaris@gmail.com>
:license: Apache License 2.0, see LICENSE for more details.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# third-party
import numpy as np

# project-level
from qudotpy import qudot


def create_ising_ham(J, h, N):
    """ Use to create an Ising Hamiltonian.

        For more info on the Ising Hamiltonian we use see
        #link

    Args:
        J: the electron interaction energy
        h: the magnetic field energy
        N: number of electrons

    Returns: Ising hamiltonian matrix
    """
    interaction_gates = [qudot.I for i in range(0, N-2)]
    interaction_gates.insert(0, qudot.X)
    interaction_gates.insert(0, qudot.X)
    interaction_mat = qudot.tensor_gate_list(interaction_gates)

    magnetic_gates = [qudot.I for i in range(0, N-1)]
    magnetic_gates.insert(0, qudot.Z)
    magnetic_mat = qudot.tensor_gate_list(magnetic_gates)

    for i in range(1, N):
        magnetic_gates[i] = qudot.Z
        magnetic_gates[i-1] = qudot.I
        magnetic_mat += qudot.tensor_gate_list(magnetic_gates)

        interaction_gates[i-1] = qudot.I
        if i+1 < N:
            interaction_gates[i+1] = qudot.X
        else:
            interaction_gates[0] = qudot.X

        interaction_mat += qudot.tensor_gate_list(interaction_gates)

    hamiltonian = (-1*J*interaction_mat) - (h * magnetic_mat)
    return hamiltonian

def solve_ground_state(H):
    """ Will solve for the ground state and energy of a hamiltonian

    We do exact diagonalization using the numpy library.

    Args:
        H: the hamiltonian matrix

    Returns:
        a tuple of a number and QuState representing the
        ground state energy and the ground state
    """
    w, v = np.linalg.eigh(H)
    ground_state_energy = w[0]
    ground_state = qudot.QuState.init_from_vector(np.asarray(v[:,0]))

    return ground_state_energy, ground_state


if __name__ == "__main__":
    ham3 = create_ising_ham(1, 1000, 3)
    energy, state = solve_ground_state(ham3)

    print("energy: %s \n state: %s" % (str(energy),
                                       state.possible_measurements()))

    ham12 = create_ising_ham(1,1,12)
    energy, state = solve_ground_state(ham12)
    print(energy)
    print(state.possible_measurements())
