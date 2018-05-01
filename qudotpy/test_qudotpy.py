# -*- coding: utf-8 -*-
"""qudot.py

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

import math
import unittest

import numpy

import qudotpy.qudot as qudot
import qudotpy.errors as qudot_errors
import qudotpy.utils as qudot_utils
import qudotpy.qulib as qulib


ONE_OVER_SQRT_2 = 1 / math.sqrt(2)

def get_column_vector(element_list):
    return numpy.array([element_list]).T

def get_row_vector(element_list):
    return numpy.array([element_list])

def vectors_equal(vec1, vec2):
    numpy.testing.assert_array_almost_equal(vec1, vec2)


class QuBitTest(unittest.TestCase):

    def setUp(self):
        self.one = qudot.QuBit("1")
        self.zero = qudot.QuBit("0")
        self.plus = qudot.QuBit("+")
        self.minus = qudot.QuBit("-")

    def tearDown(self):
        del self.one
        del self.zero
        del self.plus
        del self.minus

    def test_init(self):
        self.assertRaises(qudot_errors.InvalidQuBitError,
                          lambda : qudot.QuBit("5"))
        self.assertRaises(qudot_errors.InvalidQuBitError,
                          lambda : qudot.QuBit(4))

    def test_str(self):
        self.assertEqual(str(self.zero), "|0>")
        self.assertEqual(str(self.one), "|1>")
        self.assertEqual(str(self.plus), "|+>")
        self.assertEqual(str(self.minus), "|->")

    def test_state(self):
        amplitude = 1 / math.sqrt(2)
        tmp_zero = get_column_vector([1,0])
        tmp_one = get_column_vector([0,1])
        tmp_plus = get_column_vector([amplitude, amplitude])
        tmp_minus = get_column_vector([amplitude, -amplitude])

        vectors_equal(tmp_zero, self.zero.ket)
        vectors_equal(tmp_one, self.one.ket)
        vectors_equal(tmp_plus, self.plus.ket)
        vectors_equal(tmp_minus, self.minus.ket)

    def test_adjoint(self):
        amplitude = 1 / math.sqrt(2)
        tmp_zero = get_row_vector([1,0])
        tmp_one = get_row_vector([0,1])
        tmp_plus = get_row_vector([amplitude, amplitude])
        tmp_minus = get_row_vector([amplitude, -amplitude])

        vectors_equal(tmp_zero, self.zero.bra)
        vectors_equal(tmp_one, self.one.bra)
        vectors_equal(tmp_plus, self.plus.bra)
        vectors_equal(tmp_minus, self.minus.bra)

    def test_equals(self):
        self.assertEqual(self.one, qudot.ONE)
        self.assertEqual(self.zero, qudot.ZERO)
        self.assertEqual(self.plus, qudot.PLUS)
        self.assertEqual(self.minus, qudot.MINUS)

    def test_not_equals(self):
        self.assertNotEqual(self.one, qudot.ZERO)
        self.assertNotEqual(self.one, qudot.PLUS)
        self.assertNotEqual(self.one, qudot.MINUS)
        self.assertNotEqual(self.zero, qudot.PLUS)
        self.assertNotEqual(self.one, qudot.MINUS)
        self.assertNotEqual(self.plus, qudot.MINUS)


class QuStateTest(unittest.TestCase):

    def setUp(self):
        amplitude = 1 / math.sqrt(5)
        self.qubit_map = {
            "1000": amplitude,
            "0010": amplitude,
            "0011": amplitude,
            "1010": amplitude,
            "1011": amplitude
        }
        self.test_state = qudot.QuState(self.qubit_map)
        self.test_amplitude = amplitude

        # |0000>, |0010>, |0100>, |0110>
        self.base_vector = [.5, 0, .5, 0, .5, 0, .5j, 0,
                              0, 0, 0, 0, 0, 0, 0, 0]

        self.base_vector_real = [.5, 0, .5, 0, .5, 0, .5, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0]

        self.adj_vector = [.5, 0, .5, 0, .5, 0, -.5j, 0,
                           0, 0, 0, 0, 0, 0, 0, 0]

    def tearDown(self):
        del self.test_state
        del self.base_vector
        del self.base_vector_real
        del self.adj_vector

    def test_init_from_map(self):
        self.assertRaises(qudot_errors.InvalidQuStateError,
                          lambda : qudot.QuState(None))
        self.assertRaises(qudot_errors.InvalidQuBitError,
                          lambda : qudot.QuState({"}": 1}))

        # make map with even states in 4D Hilbert space
        qubit_map = {}
        for i in range(0, 8):
            if not i % 2:
                if i == 6:
                    qubit_map[qudot_utils.int_to_bit_str(i, 4)] = 0.5j
                else:
                    qubit_map[qudot_utils.int_to_bit_str(i, 4)] = 0.5

        qu_state = qudot.QuState(qubit_map)

        column_vector = get_column_vector(self.base_vector)
        row_vector = get_row_vector(self.adj_vector)
        vectors_equal(column_vector, qu_state.ket)
        vectors_equal(row_vector, qu_state.bra)
        self.assertTrue(4 == qu_state.num_qubits)
        self.assertTrue(2**4 == qu_state.hilbert_dimension)

    def test_init_from_list(self):
        self.assertRaises(qudot_errors.InvalidQuStateError,
                          lambda : qudot.QuState([]))
        self.assertRaises(qudot_errors.InvalidQuBitError,
                          lambda : qudot.QuState.init_from_state_list(["{"]))

        qu_state_lst = ["0000", "0010", "0100", "0110"]
        qu_state = qudot.QuState.init_from_state_list(qu_state_lst)
        column_vector = get_column_vector(self.base_vector_real)
        row_vector = get_row_vector(self.base_vector_real)
        vectors_equal(column_vector, qu_state.ket)
        vectors_equal(row_vector, qu_state.bra)
        self.assertTrue(4 == qu_state.num_qubits)
        self.assertTrue(2**4 == qu_state.hilbert_dimension)

    def test_init_superposition(self):
        qu_state = qudot.QuState.init_superposition(4)
        self.assertTrue(2**4 == qu_state.hilbert_dimension)
        self.assertTrue(4 == qu_state.num_qubits)

        amplitude = .25
        vector = []
        for i in range(0, 16):
            vector.append(amplitude)

        column_vector = get_column_vector(vector)
        row_vector = get_row_vector(vector)
        vectors_equal(column_vector, qu_state.ket)
        vectors_equal(row_vector, qu_state.bra)

    def test_init_from_vector(self):
        vector = [.707, 0, 0, .707]
        column_vector = get_column_vector(vector)
        row_vector = get_row_vector(vector)
        # must be column vector
        self.assertRaises(ValueError,
                          lambda : qudot.QuState.init_from_vector(row_vector))
        qu_state = qudot.QuState.init_from_vector(column_vector)
        self.assertTrue(qu_state.num_qubits == 2)
        self.assertTrue(qu_state.hilbert_dimension == 2**2)
        vectors_equal(column_vector,qu_state.ket)
        vectors_equal(row_vector, qu_state.bra)

    def test_init_zeros(self):
        self.assertTrue(str(qudot.QuState.init_zeros(2)), "1|00>")
        self.assertTrue(str(qudot.QuState.init_zeros(3)), "1|000>")
        self.assertTrue(str(qudot.QuState.init_zeros(4)), "1|0000>")
        self.assertTrue(str(qudot.QuState.init_zeros(6)), "1|000000>")
        self.assertTrue(str(qudot.QuState.init_zeros(9)), "1|000000000>")

    def test_str(self):
        vector = [0.707, 0, 0, 0.707]
        column_vector = get_column_vector(vector)
        qu_state = qudot.QuState.init_from_vector(column_vector)
        dirac_str = "0.707|00> + 0.707|11>"
        self.assertEqual(dirac_str, str(qu_state))

        vector = [0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5]
        column_vector = get_column_vector(vector)
        qu_state = qudot.QuState.init_from_vector(column_vector)
        dirac_str = "0.5|000> + 0.5|001> + 0.5|011> + 0.5|111>"
        self.assertEqual(dirac_str, str(qu_state))

    def test_possible_measurements(self):
        possible_measurements = self.test_state.possible_measurements()

        self.assertTrue("|0010>" in possible_measurements)
        self.assertTrue("|0011>" in possible_measurements)
        self.assertTrue("|1000>" in possible_measurements)
        self.assertTrue("|1010>" in possible_measurements)
        self.assertTrue("|1011>" in possible_measurements)
        self.assertFalse("|1111>" in possible_measurements)

        total_probablity = 0
        for state in possible_measurements:
            total_probablity += possible_measurements[state]

        self.assertAlmostEqual(total_probablity, 1, places=6)

        possible_qubit_value = self.test_state.possible_measurements(2)
        self.assertTrue("|0>" in possible_qubit_value)
        self.assertFalse("|1>" in possible_qubit_value)

        possible_qubit_value = self.test_state.possible_measurements(1)
        prob_one = qudot.measurement_probability(self.test_amplitude) * 3
        prob_zero = qudot.measurement_probability(self.test_amplitude) * 2
        self.assertAlmostEqual(prob_one, possible_qubit_value["|1>"])
        self.assertAlmostEqual(prob_zero, possible_qubit_value["|0>"])

    def test_measure(self):
        possible_measurements = self.test_state.possible_measurements()
        measurement = self.test_state.measure()
        self.assertTrue(measurement in possible_measurements)
        int_result = qudot_utils.dirac_str_to_int(measurement)
        self.assertTrue(self.test_state.ket[int_result][0] == 1)
        for possible_measurement in possible_measurements:
            if possible_measurement != measurement:
                index = qudot_utils.dirac_str_to_int(possible_measurement)
                self.assertTrue(self.test_state.ket[index][0] == 0)

    def test_equals(self):
        test_list = []
        for key in self.qubit_map:
            test_list.append(key)
        test_state1 = qudot.QuState.init_from_state_list(test_list)
        self.assertEqual(test_state1, self.test_state)

    def test_not_equals(self):
        test_state2 = qudot.QuState.init_superposition(4)
        self.assertNotEqual(test_state2, self.test_state)

    def test_apply_gate(self):
        test_input = qudot.QuState.init_zeros(2)
        output = qudot.QuState({"00": .5, "10": .5, "01": .5, "11":-.5})
        test_input.apply_gate(qudot.H, [1])
        test_input.apply_gate(qudot.CNOT)
        test_input.apply_gate(qudot.H, [1])
        self.assertEqual(test_input, output)

        test_input = qudot.QuState.init_zeros(2)
        output = qudot.QuState({"00": 1/math.sqrt(2), "11": 1/math.sqrt(2)})
        test_input.apply_gate(qudot.H, [1])
        test_input.apply_gate(qudot.CNOT)
        self.assertEqual(test_input, output)



class QuGateTest(unittest.TestCase):

    def test_equals(self):
        self.assertTrue(qudot.X == qudot.X)
        self.assertTrue(qudot.Z == qudot.Z)
        self.assertTrue(qudot.Y == qudot.Y)
        self.assertTrue(qudot.H == qudot.H)
        self.assertTrue(qudot.X != qudot.Y)
        self.assertTrue(qudot.Y != qudot.X)
        self.assertTrue(qudot.H != qudot.Z)
        self.assertTrue(qudot.Y != qudot.Z)

    def test_init_from_str(self):
        self.assertTrue(qudot.X == qudot.QuGate.init_from_str("0 1; 1 0"))
        self.assertTrue(qudot.Y != qudot.QuGate.init_from_str("1 0;0 -1 "))
        self.assertTrue(qudot.H == qudot.QuGate.init_from_str(
            "1 1; 1 -1", 1/math.sqrt(2)))
        # not square matrix
        self.assertRaises(qudot_errors.InvalidQuGateError,
                          lambda : qudot.QuGate.init_from_str("1 0 0; 0 1 0"))
        # not unitary
        self.assertRaises(qudot_errors.InvalidQuGateError,
                          lambda : qudot.QuGate.init_from_str("1 1; 1 -1"))

    def test_init_from_multiplication(self):
        # use well known quantum circuit identities for test
        # see Nielsan & Chang pg 177
        qu_gates = [qudot.H, qudot.X, qudot.H]
        self.assertEqual(qudot.Z, qudot.QuGate.init_from_mul(qu_gates))

        qu_gates = [qudot.H, qudot.Z, qudot.H]
        self.assertEqual(qudot.X, qudot.QuGate.init_from_mul(qu_gates))

        qu_gates = [qudot.H, qudot.Y, qudot.H]
        minus_Y = qudot.QuGate(qudot.Y.matrix, -1)
        self.assertEqual(minus_Y, qudot.QuGate.init_from_mul(qu_gates))

        self.assertRaises(qudot_errors.InvalidQuGateError,
                          lambda : qudot.QuGate.init_from_mul([]))

        qu_gates = [qudot.X, qudot.CNOT]
        self.assertRaises(qudot_errors.InvalidQuGateError,
                          lambda : qudot.QuGate.init_from_mul(qu_gates))

    def test_mul(self):
        self.assertEqual(qudot.Z, qudot.H * qudot.X * qudot.H)
        self.assertEqual(qudot.X, qudot.H * qudot.Z * qudot.H)
        self.assertEqual(-1 * qudot.Y, qudot.H * qudot.Y * qudot.H)


    def test_init_from_tensor_product(self):
        X_X = qudot.QuGate.init_from_str("0 0 0 1;"
                                         "0 0 1 0;"
                                         "0 1 0 0;"
                                         "1 0 0 0")

        X_Z = qudot.QuGate.init_from_str("0 0 1 0;"
                                         "0 0 0 -1;"
                                         "1 0 0 0;"
                                         "0 -1 0 0")

        H_H = qudot.QuGate.init_from_str("1 1 1 1;"
                                         "1 -1 1 -1;"
                                         "1 1 -1 -1;"
                                         "1 -1 -1 1",
                                         multiplier=0.5)

        gates = [qudot.X, qudot.X]
        self.assertEqual(X_X, qudot.QuGate.init_from_tensor_product(gates))

        gates = [qudot.X, qudot.Z]
        self.assertEqual(X_Z, qudot.QuGate.init_from_tensor_product(gates))

        gates = [qudot.H, qudot.H]
        self.assertEqual(H_H, qudot.QuGate.init_from_tensor_product(gates))

        gates = [qudot.H, qudot.H, qudot.H, qudot.H, qudot.H]
        qudot.QuGate.init_from_tensor_product(gates)

    def test_init_control_gate(self):
        cnot = qudot.QuGate.init_control_gate(qudot.X)
        self.assertEqual(cnot, qudot.CNOT)
        self.assertNotEqual(cnot, qudot.X)

        cz = qudot.QuGate.init_control_gate(qudot.Z)
        cz_manual = qudot.QuGate.init_from_str("1 0 0 0;"
                                               "0 1 0 0;"
                                               "0 0 1 0;"
                                               "0 0 0 -1")
        self.assertEqual(cz, cz_manual)

        cy = qudot.QuGate.init_control_gate(qudot.Y)
        cy_manual = qudot.QuGate.init_from_str("1 0 0 0;"
                                               "0 1 0 0;"
                                               "0 0 0 -1j;"
                                               "0 0 1j 0")
        self.assertEqual(cy, cy_manual)

        self.assertRaises(qudot_errors.InvalidQuGateError,
                          lambda : qudot.QuGate.init_control_gate(qudot.X, control_qubit=1, target_qubit=2, num_qubits=1))

        self.assertRaises(qudot_errors.InvalidQuGateError,
                          lambda : qudot.QuGate.init_control_gate(qudot.X, control_qubit=1, target_qubit=1, num_qubits=3))

        self.assertRaises(qudot_errors.InvalidQuGateError,
                          lambda : qudot.QuGate.init_control_gate(qudot.X, control_qubit=4, target_qubit=1, num_qubits=2))

        self.assertRaises(qudot_errors.InvalidQuGateError,
                          lambda : qudot.QuGate.init_control_gate(qudot.X, control_qubit=4, target_qubit=5, num_qubits=3))

    def test_init_phase_gate(self):
        r0 = qudot.QuGate.init_phase_gate(0)
        r1 = qudot.QuGate.init_phase_gate(1)
        r2 = qudot.QuGate.init_phase_gate(2)
        r3 = qudot.QuGate.init_phase_gate(3)

        self.assertEqual(r0, qudot.I)
        self.assertEqual(r1, qudot.Z)
        self.assertEqual(r2, qudot.S)
        self.assertEqual(r3, qudot.T)

        c_r1 = qudot.QuGate.init_control_gate(r1)
        c_r2 = qudot.QuGate.init_control_gate(r2)
        c_r3 = qudot.QuGate.init_control_gate(r3)

        self.assertEqual(c_r1, qudot.QuGate.init_control_gate(r1, control_qubit=2, target_qubit=1))
        self.assertEqual(c_r2, qudot.QuGate.init_control_gate(r2, control_qubit=2, target_qubit=1))
        self.assertEqual(c_r3, qudot.QuGate.init_control_gate(r3, control_qubit=2, target_qubit=1))


# Module Level test cases
class QuDotTest(unittest.TestCase):

    def test_gate_operations(self):
        result = qudot.apply_gate(qudot.X, qudot.ZERO)
        self.assertEqual(result, qudot.ONE)

        result = qudot.apply_gate(qudot.X, qudot.ONE)
        self.assertEqual(result, qudot.ZERO)

        result = qudot.apply_gate(qudot.X, qudot.PLUS)
        self.assertEqual(result, qudot.PLUS)

        result = qudot.apply_gate(qudot.Z, qudot.ZERO)
        self.assertEqual(result, qudot.ZERO)

        result = qudot.apply_gate(qudot.Z, qudot.ONE)
        vectors_equal(result.ket, qudot.ONE.ket * (-1))

        result = qudot.apply_gate(qudot.H, qudot.ZERO)
        self.assertEqual(result, qudot.PLUS)

        result = qudot.apply_gate(qudot.H, qudot.ONE)
        self.assertEqual(result, qudot.MINUS)

        result = qudot.apply_gate(qudot.H, qudot.PLUS)
        self.assertEqual(result, qudot.ZERO)

        result = qudot.apply_gate(qudot.H, qudot.MINUS)
        self.assertEqual(result, qudot.ONE)


class QuLibTest(unittest.TestCase):

    def test_ripple_carry(self):
        q2 = qudot.QuState.init_from_state_list(["10"])
        q1 = qudot.QuState.init_from_state_list(["01"])
        r1 = qudot.QuState.init_from_state_list(["011"])
        self.assertEqual(r1, qulib.ripple_adder(q2, q1))

        q3 = qudot.QuState.init_from_state_list(["11"])
        r2 = qudot.QuState.init_from_state_list(["110"])
        self.assertEqual(r2, qulib.ripple_adder(q3, q3))

if __name__ == "__main__":
    unittest.main()
