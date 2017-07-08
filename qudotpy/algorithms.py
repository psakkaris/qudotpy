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
    h_list = [qudot.I for x in range(0, num_qubits)]
    qft_matrices = []
    for q in range(1, num_qubits+1):
        sub_matrices = []
        # create the Haddamard part
        h_list[q-1] = qudot.H
        haddamard = qudot.QuGate.init_from_tensor_product(h_list)
        sub_matrices.insert(0, haddamard)
        h_list[q-1] = qudot.I

        # create the control rotations
        i = q + 1
        r = 2
        while i <= num_qubits:
            phase_gate = qudot.QuGate.init_phase_gate(r)
            control = qudot.QuGate.init_control_gate(phase_gate, i, q, num_qubits)
            sub_matrices.insert(0, control)
            i += 1
            r += 1

        qft_matrices.insert(0, qudot.QuGate.init_from_mul(sub_matrices))

    for q in range(1, int(num_qubits / 2) + 1):
        swap = qudot.QuGate.init_swap_gate(q, num_qubits - q + 1, num_qubits)
        qft_matrices.insert(0, swap)

    return qudot.QuGate.init_from_mul(qft_matrices)


def inverse_qft(num_qubits):
    qft_gate = qft(num_qubits)
    return qudot.QuGate(qft_gate.dagger)


if __name__ == "__main__":
    qft3 = qft(3)
    rows = qft3.matrix.shape[0]
    for row in range(0, rows):
        row_txt = ""
        for col in range(0, rows):
            row_txt += "%s\t" % qft3.matrix.item((row, col))
        print(row_txt + "\n")
