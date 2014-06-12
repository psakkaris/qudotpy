__author__ = 'psakkaris'

import unittest
import qudot
import qudot_errors
import numpy
import math

ONE_OVER_SQRT_2 = .707106

def get_column_vector(element_list):
    return numpy.array([element_list]).T

def get_row_vector(element_list):
    return numpy.array([element_list])


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

        self.assertTrue(numpy.array_equal(tmp_zero, self.zero.state))
        self.assertTrue(numpy.array_equal(tmp_one, self.one.state))
        self.assertTrue(numpy.array_equal(tmp_plus, self.plus.state))
        self.assertTrue(numpy.array_equal(tmp_minus, self.minus.state))

    def test_adjoint(self):
        amplitude = 1 / math.sqrt(2)
        tmp_zero = get_row_vector([1,0])
        tmp_one = get_row_vector([0,1])
        tmp_plus = get_row_vector([amplitude, amplitude])
        tmp_minus = get_row_vector([amplitude, -amplitude])

        self.assertTrue(numpy.array_equal(tmp_zero, self.zero.adjoint))
        self.assertTrue(numpy.array_equal(tmp_one, self.one.adjoint))
        self.assertTrue(numpy.array_equal(tmp_plus, self.plus.adjoint))
        self.assertTrue(numpy.array_equal(tmp_minus, self.minus.adjoint))

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
    pass

class QuGateTest(unittest.TestCase):
    pass

# Module Level test cases
class QuDotTest(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
