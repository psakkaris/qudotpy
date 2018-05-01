# -*- coding: utf-8 -*-
"""qudotpy.qlib

Module to for common quantum library methods

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

from qudotpy import qudot
from qudotpy import utils
from qudotpy import algorithms


def carry(start, qu_state):
    """
    Quantum carry operation
    :param start: The starting qubit to apply the carry
    :param qu_state: a QuState
    :return: altered QuState
    """
    qu_state.apply_toffoli_gate(start + 1, start + 2, start + 3)
    qu_state.apply_control_gate(qudot.X, start + 1, start + 2)
    qu_state.apply_toffoli_gate(start, start + 2, start + 3)


def qsum(start, qu_state):
    """
    Quanutm sum operation
    :param start: The starting qubit to apply the sum
    :param qu_state: a QuState
    :return: altered QuState
    """
    qu_state.apply_control_gate(qudot.X, start + 1, start + 2)
    qu_state.apply_control_gate(qudot.X, start, start + 2)


def rev_carry(start, qu_state):
    """
    The reverse carry operation, or the conjugate. Undoes what carry method does
    :param start: The starting qubit to apply the reverse carry
    :param qu_state: altered QuState
    :return:
    """
    qu_state.apply_toffoli_gate(start, start + 2, start + 3)
    qu_state.apply_control_gate(qudot.X, start + 1, start + 2)
    qu_state.apply_toffoli_gate(start + 1, start + 2, start + 3)


def ripple_adder(qu_state1, qu_state2):
    """
    Adds the two QuStates and returns their sum using ripple_carry algorithm. The two states must have the same
    number of qubits. Returns a new QuState that is their sum.
    :param qu_state1: QuState
    :param qu_state2: QuState
    :return:  qu_state1 + qu_state2
    """
    if qu_state1.num_qubits != qu_state2.num_qubits:
        raise ValueError("number of qubits do not match for state")

    state1 = utils.state_to_bit_str(qu_state1)
    state2 = utils.state_to_bit_str(qu_state2)
    length = qu_state1.num_qubits

    add_state = []
    for i in range(0, length):
        add_state.append("0")
        add_state.append(state1[length-i-1])
        add_state.append(state2[length-i-1])

    add_state.append("0")
    add_str = "".join(add_state)
    qu_state = qudot.QuState.init_from_state_list([add_str])

    qubit = 1

    while qubit < qu_state.num_qubits:
        carry(qubit, qu_state)
        qubit += 3

    qubit -= 3
    qu_state.apply_control_gate(qudot.X, qubit+1, qubit+2)
    qsum(qubit, qu_state)

    qubit -= 3
    while qubit >= 1:
        rev_carry(qubit, qu_state)
        qsum(qubit, qu_state)
        qubit -= 3

    results_bits = utils.state_to_bit_str(qu_state)
    start = qu_state.num_qubits - 2
    results_bit_list = [results_bits[start+1]]

    while start > 0:
        results_bit_list.append(results_bits[start])
        start -= 3

    result_str = "".join(results_bit_list)
    return qudot.QuState.init_from_state_list([result_str])


def qft_adder(qu_state1, qu_state2):
    num_qubits = qu_state1.num_qubits
    if qu_state1.num_qubits != qu_state2.num_qubits:
        raise ValueError("number of qubits do not match for state")

    qftn = algorithms.qft(num_qubits)
    qu_state1.apply_gate(qftn)
    state2 = utils.state_to_bit_str(qu_state2)

    for i in range(0, num_qubits):
        qubit = num_qubits - i
        for b in range(1+i, num_qubits+1):
            r = b-i
            if state2[b-1] == "1":
                phase_gate = qudot.QuGate.init_phase_gate(r)
                qu_state1.apply_gate(phase_gate, [qubit])

    iqftn = algorithms.inverse_qft(num_qubits)
    qu_state1.apply_gate(iqftn)
