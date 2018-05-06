# -*- coding: utf-8 -*-
"""qudotpy.qudot

Description goes here...

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

# project-level
from qudotpy import qudot


def qft(num_qubits):
    """Implementation of the quantum fourier transform.

    Args:
        num_qubits: the number of qubits the qft will apply to
    """
    h_list = [qudot.I for x in range(0, num_qubits)]
    qft_matrices = []
    for qubit in range(1, num_qubits+1):
        sub_matrices = []
        # create the Haddamard part
        h_list[qubit-1] = qudot.H
        haddamard = qudot.QuGate.init_from_tensor_product(h_list)
        sub_matrices.insert(0, haddamard)
        h_list[qubit-1] = qudot.I

        # create the control rotations
        i = qubit + 1
        phase = 2
        while i <= num_qubits:
            phase_gate = qudot.QuGate.init_phase_gate(phase)
            control = qudot.QuGate.init_control_gate(phase_gate, i, qubit, num_qubits)
            sub_matrices.insert(0, control)
            i += 1
            phase += 1

        qft_matrices.insert(0, qudot.QuGate.init_from_mul(sub_matrices))

    for qubit in range(1, int(num_qubits / 2) + 1):
        swap = qudot.QuGate.init_swap_gate(qubit, num_qubits - qubit + 1, num_qubits)
        qft_matrices.insert(0, swap)

    return qudot.QuGate.init_from_mul(qft_matrices)


def inverse_qft(num_qubits):
    """Implementation of the inverse quanutm fourier transform.

    Args:
        num_qubits: the number of qubits the iqft will apply to.
    """
    qft_gate = qft(num_qubits)
    return qudot.QuGate(qft_gate.dagger)


if __name__ == "__main__":
    QFT3 = qft(3)
    ROWS = QFT3.matrix.shape[0]
    for row in range(0, ROWS):
        row_txt = ""
        for col in range(0, ROWS):
            row_txt += "%s\t" % QFT3.matrix.item((row, col))
        print(row_txt + "\n")
