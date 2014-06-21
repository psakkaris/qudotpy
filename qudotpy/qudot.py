# -*- coding: utf-8 -*-
"""qudotpy.qudot

Description goes here...

:copyright: Copyright (C) 2014 QuDot, Inc. | Copyright (C) 2014 Perry Sakkaris <psakkaris@gmail.com>
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

import numpy as np
import math
import random
import qudot_utils
import qudot_errors

class QuBaseState(object):
    """Common properties of quantum states.

    This class is not intended to be used directly, but to be
    subclassed as in QuBit and QuState. The subclass should defined
    a private variable _state which is a column vector representation
    of the quantum state. Then the properties state and adjoint will
    automatically inherited.
    """

    @property
    def state(self):
        """Column vector representation of the quantum state"""
        return self._state

    @property
    def adjoint(self):
        """The Hermitian Conjugate of the state a.k.a row vector"""
        return np.conjugate(self._state.T)

    def __eq__(self, other):
        """Considered equal if all elements of a state are within threshold"""
        #TODO: choose and configure project wide tolerance value
        equal = False
        if hasattr(other, "state"):
            equal = np.allclose(self.state, other.state)

        return equal

    def __ne__(self, other):
        """Logically opposite of __eq__"""
        #TODO: choose and configure project wide tolerance value
        return not self.__eq__(other)



class QuBit(QuBaseState):
    """A two level quantum system.

    The standard two level systems in quantum computing are
    0,1: the computational basis
    +,-: the Hadamard basis
    We use numpy vectors to represent these qubits and display them
    using Dirac notation

    Attributes:
        column_vector: qubit represented as a column vector
        row_vector: qubit represented as a row vector

    Constants:
        ZERO: string representation of the 0 qubit
        ONE: string representation of the 1 qubit
        PLUS: string representation of the + qubit
        MINUS: string representation of the - qubit
        DISPLAY_MAP: maps string representation of qubit to
                     Dirac notation string
    """


    ZERO = "0"
    ONE = "1"
    PLUS = "+"
    MINUS = "-"
    DISPLAY_MAP = {ZERO: "|0>", ONE: "|1>",
                   PLUS: "|+>", MINUS: "|->"}

    def __init__(self, qubit_str):
        """Create a QuBit object from its string representation.

        Args:
            qubit_str: one of "1", "0", "+", "-"

        Raises:
            InvalidQubitError: didn't recognize the qubit string
        """
        if qubit_str == QuBit.ZERO:
            self._state = np.array([[1], [0]])
        elif qubit_str == QuBit.ONE:
            self._state = np.array([[0], [1]])
        elif qubit_str == QuBit.PLUS:
            self._state = np.array([[1], [1]]) * 1 / np.sqrt(2)
        elif qubit_str == QuBit.MINUS:
            self._state = np.array([[1], [-1]]) * 1 / np.sqrt(2)
        else:
            message = ("A qubit must be one of the strings %s, %s, %s or %s"
                       % (QuBit.ZERO, QuBit.ONE, QuBit.PLUS, QuBit.MINUS))
            raise qudot_errors.InvalidQuBitError(message)

        self._state_str = qubit_str

    def __str__(self):
        """Dirac notation of qubit """
        return QuBit.DISPLAY_MAP[self._state_str]


class QuState(QuBaseState):
    """A general quantum state in the computational basis.

    Represents a quantum state in any Hilbert dimension. The underlying
    state is stored as a column vector.
    We display the state using Dirac Notation.
    ex: 1/sqrt(2)|00> + 1/sqrt(2)|11>
    You can initialize a state in the Hadamard basis but it
    will be stored and displayed in the computational basis.

    Attributes:
        state: the column vector of the state
        hilbert_dimension: the dimensionality of the Hilbert Space
    """


    @property
    def hilbert_dimension(self):
        """The dimensionality of the associated Hilbert space"""
        return self._hilbert_dimension

    @property
    def num_qubits(self):
        """The number of qubits needed to represent the QuState"""
        return self._num_qubits

    def __init__(self, state_map):
        """Create a QuState from a map.

        Args:
            state_map: a map where keys are the bit string of the state
                       and the values are its probability amplitudes,
                       example:
                           "001" => 1/sqrt(2)
                           "111" => 1/sqrt(2)
                           Will be the state
                           1/sqrt(2)|001> + 1/sqrt(2)|111>
        Raises:
            InvalidQuStateError: if map is not supplied
            InvalidQuBitError: if we encounter a string that does
                               not map to a qubit like "}"
        """
        if state_map:
            for state_tuple in state_map.items():
                state_bit_str = state_tuple[0]
                amplitude = state_tuple[1]
                tmp = None
                for bit in state_bit_str[::-1]:
                    if tmp is None:
                        tmp = QuBit(bit).state
                    else:
                        tmp = np.kron(QuBit(bit).state, tmp)

                if hasattr(self, "_state"):
                    self._state += (tmp * amplitude)
                else:
                    self._state = (tmp * amplitude)

                self._num_qubits = int(math.log(self._state.size, 2))
                self._hilbert_dimension = self._state.size
        else:
            message = ("you must provide a map with your states as keys and"
                       "and amplitudes as values. "
                       "Ex: {\"00\":1/sqrt(2), \"11\":.5/sqrt(2)}")
            raise qudot_errors.InvalidQuStateError(message)

    @classmethod
    def init_from_state_list(cls, state_list):
        """Create an equiprobable superposition states

        This will create a QuState that is a superposition of the
        states specified in argument and will assign the same
        probability amplitude to all states.
        Example: ["00","11"] => 1/sqrt(2)|00> + 1/sqrt(2)|11>

        Args:
            state_list: a list of strings representing qubit states

        Raises:
            InvalidQuStateError: if state_list is not supplied
            InvalidQuBitError: if we encounter a string that does
                               not map to a qubit like "}"
        """
        state_map = {}
        if state_list:
            amplitude = 1 / math.sqrt(len(state_list))
            for state in state_list:
                state_map[state] = amplitude

        return cls(state_map)

    @classmethod
    def init_superposition(cls, dimension):
        """Create an equiprobable superposition of ALL states in a space

        This method will create a superposition of all basis states in
        the Hilbert space with a given dimension.
        Example: for a 3 dimensional Hilbert sapce this will create
        |000> + |001> + |010> + |011> + |100> + |101> + |110> + |111>
        all with probability amplitude 1/sqrt(8)

        Args:
            dimension: integer representing the size of the space
        """
        state_map = {}
        if dimension:
            num_states = 2**dimension
            amplitude = 1 / math.sqrt(num_states)
            for state in range(0, num_states):
                bit_str = bin(state)[2:]
                bit_str = "0" * (dimension - len(bit_str)) + bit_str
                state_map[bit_str] = amplitude

        return cls(state_map)

    @classmethod
    def init_from_vector(cls, vector):
        """Create a QuState from a column vector.

        This method will create a QuState given a raw column vector as input.
        Example: [[.707],
                  [0],
                  [0],
                  [.707]]
        Will create the QuState .707|00> + .707|11>

        Args:
            vector: a vector representing a state in Hilbert Space
                    Note that this must have size length 2^n or else
                    it is considered invalid.
        """
        state_map = {}
        vector_len = len(vector)
        if vector_len % 2:
            message = ("Vector representations of quantum states must be a "
                      "power of 2. Your vector has length %s" % str(vector_len))
            raise ValueError(message)
        if len(vector):
            dimensionality = int(math.log(len(vector), 2))
            i = 0
            for element in vector:
                if element:
                    bit_str = qudot_utils.int_to_bit_str(i, dimensionality)
                    state_map[bit_str] = element[0]
                i += 1

        return cls(state_map)

    @classmethod
    def init_zeros(cls, num_bits):
        """ Initialized a state with all zeros

        examples: |00>, |00000000>, |000000000000000000000000>
        Args:
            num_bits: the number of zeros you want
        """
        bit_str = "0" * num_bits
        return cls({bit_str: 1})

    def __str__(self):
        """Dirac notation of qubit """
        my_str = []
        index = 0
        for element in self.state:
            if element:
                if my_str:
                    my_str.append(" + ")
                my_str.append(str(element[0]))
                bit_str = qudot_utils.int_to_dirac_str(
                    index,
                    self._num_qubits)
                my_str.append(bit_str)
            index += 1
        return "".join(my_str)

    def _collapse(self, state_str):
        """collapses the state to the specified state_str"""
        state_index = qudot_utils.dirac_str_to_int(state_str)
        index = 0
        for element in self._state:
            if index == state_index:
                element[0] = 1
            else:
                element[0] = 0

            index += 1

    def possible_measurements(self, qubit_index=-1):
        """Returns all possible measurements and their probability.

        The return value will be a map where the key is a string of the
        possible state and the value is its probability of outcome. You
        can return the measurement probability of a specific qubit by
        using qubit_index param. Example, the third qubit of the state
        1/sqrt(2)|0010> + 1/sqrt(2)|1110> can be 1 with probability 1

        Args:
            qubit_index: use if you want to know the possible
                         measurements of a specific qubit. First
                         qubit is 1.

        Raises:
            ValueError: if you specify a qubit that is out of range
        """
        states_map = {}
        index = 0
        for element in self.state:
            if element:
                probablility = measurement_probability(element[0])
                dirac_str = qudot_utils.int_to_dirac_str(
                    index,
                    self._num_qubits)
                states_map[dirac_str] = probablility
            index += 1

        if qubit_index >= 0:
            if qubit_index > self._num_qubits:
                raise ValueError("qubit_index=%s is out of range" %
                                 str(qubit_index))
            qubit_map = {}
            for dirac_state in states_map:
                state_index = qudot_utils.dirac_str_to_int(dirac_state)
                amplitude = self._state[state_index][0]
                qubit = qudot_utils.DIRAC_STR % dirac_state[qubit_index]
                probablility = measurement_probability(amplitude)
                if qubit_map.has_key(qubit):
                    old_probability = qubit_map[qubit]
                    qubit_map[qubit] = old_probability + probablility
                else:
                    qubit_map[qubit] = probablility

            return qubit_map
        else:
            return states_map

    def measure(self):
        """Measure the QuState

        Measurement of the state will result in one of the possible outcomes
        with the given probability. Also, the state WILL COLLAPSE to the
        outcome so this changes the state. The probability outcomes can be
        checked by having an ensemble of the same states and performing a
        measurement on each one. As the ensemble gets larger the probable
        outcomes will match the probability amplitudes magnitude squared
        """
        rand = random.random()
        possibilities_map = self.possible_measurements()
        start = 0
        for possibility in possibilities_map:
            probability = possibilities_map[possibility]
            if start <= rand <= start + probability:
                self._collapse(possibility)
                return possibility
            start += probability

    def apply_gate(self, qu_gate, qubit_list=None):
        """ Apply a QuGate to the entire state or a qubit

        The supplied QuGate will be "scaled" to the appropriate dimension.
        For example lets say you have a state |0100> and you want to
        apply the X gate to the entire state. Then you can pass in the
        predefined qudot.X QuGate and it will be tensored with itself
        until the dimension matches that of the state.
        You can also specify a specific qubit you want to apply the gate to.
        Qubit indexing starts from one, for example, the second qubit
        of |0100> is 1.

        Args:
            qu_gate: the QuGate you want to apply
            qubit: optional. Specifies a list of qubit index you want to
                   apply the QuGate to. If not specified, the QuGate will
                   be applied to the entire state

        Raise:
            ValueError: if you try to apply to an out of bounds qubit index
        """
        if not qubit_list: qubit_list = []
        dimension = qu_gate.matrix.shape[0]
        qu_gate_list = []
        if qubit_list:
            eye_gate = QuGate(np.asmatrix(np.eye(2)))
            for bit in range(1, self._num_qubits + 1):
                if bit in qubit_list:
                    qu_gate_list.append(qu_gate)
                else:
                    qu_gate_list.append(eye_gate)

        else:
            qu_gate_list.append(qu_gate)
            while dimension < self._hilbert_dimension:
                qu_gate_list.append(qu_gate)
                dimension *= 2

        final_gate = QuGate.init_from_tensor_product(qu_gate_list)
        # make sure dimensions are same after this
        if final_gate.matrix.shape[0] != self._hilbert_dimension:
            raise RuntimeError("gate of dimension %s cannot be applied to"
                               "state with dimension %s" %
                               (final_gate.matrix.shape[0],
                                self._hilbert_dimension))

        self._state = np.asarray(final_gate.matrix * self._state)


class QuGate(object):
    """A general quantum gate.

    QuGate is a representation of a quantum gate using numpy matrices. We
    check that the gate is unitary upon initialization

    Attributes:
        matrix: the matrix representation of the gate
        dagger: the Hermitian (dagger) of the gate
    """

    @property
    def matrix(self):
        return self._matrix

    @property
    def dagger(self):
        return self._matrix.H

    def __eq__(self, other):
        """Returns true if all matrix elements of other are close """
        #TODO: choose and configure project wide tolerance value
        equals = False
        if hasattr(other, "matrix"):
            equals = np.allclose(self.matrix, other.matrix)

        return equals

    def __ne__(self, other):
        """Logically opposite of __eq__(other) """
        return not self.__eq__(other)

    def __init__(self, matrix, multiplier=-1):
        """Create a quantum gate from a numpy matrix.

        Note that if the matrix dtype is not complex it will be converted
        to a complex matrix.

        Args:
            matrix: a complex typed numpy matrix
            multiplier: if supplied all the elements of the matrix will
                        be multiplied by the multiplier

        Raises:
            InvalidQuGateError: if the gate is not unitary
        """

        if matrix.dtype == np.dtype('complex'):
            self._matrix = matrix
        else:
            self._matrix = matrix.astype('complex')

        shape = self._matrix.shape
        if shape[0] != shape[1]:
            raise qudot_errors.InvalidQuGateError("Gate is not a square matrix")

        if multiplier > 0:
            self._matrix = self._matrix * multiplier

        #TODO: choose and configure project wide tolerance value
        is_unitary = np.allclose((self._matrix.H * self._matrix).real,
            np.eye(shape[0]))
        if not is_unitary:
            raise qudot_errors.InvalidQuGateError("Gate is not unitary")

    @classmethod
    def init_from_str(cls, matrix_str, multiplier=-1):
        """Create a quantum gate from a string.

        The string of the quantum gate should be rows separated by
        semi-colons (;) and elements of a row separated by spaces
        Example: the Pauli-X gate would be "0 1; 1 0"
                 the Pauli-Y gate would be "0 1j; -1j 0)

        Args:
            matrix_str: the string representation of the quantum gate
            multiplier: if supplied all the elements of the matrix will
                        be multiplied by the multiplier

        Raises:
            InvalidQuGateError: if the gate is not unitary
        """
        matrix = np.matrix(matrix_str, dtype='complex')
        return QuGate(matrix, multiplier)

    @classmethod
    def init_from_mul(cls, qu_gates):
        """ Create a quantum gate by multiplying existing gates

        The multiplication will be done from left to right starting zeroth
        element of the qu_gates list. For example:
        qu_gates = [H, X, H] will be H * X * H

        Args:
            qu_gates: a list of QuGates to multiply

        Raises:
            InvalidQuGateError: if the result of the multiplication is
                                not unitary or if the shape of the
                                matrices does not match
        """
        if qu_gates:
            matrix = qu_gates[0].matrix
            for i in range(1, len(qu_gates)):
                if matrix.shape != qu_gates[i].matrix.shape:
                    message = "The matrices supplied have different shapes"
                    raise qudot_errors.InvalidQuGateError(message)

                matrix = matrix * qu_gates[i].matrix

            return QuGate(matrix)
        else:
            raise qudot_errors.InvalidQuGateError("No gates specified")

    @classmethod
    def init_from_tensor_product(cls, qu_gates):
        """ Create a quantum gate by the tensor product of other gates

        The tensor product will be done from left to right starting zeroth
        element of the qu_gates list. For example:
        qu_gates = [H, X, H] will be H tensor X tensor H

        Args:
            qu_gates: a list of QuGates to tensor

        Raises:
            InvalidQuGateError: if the result is not unitary
        """
        if qu_gates:
            matrix = qu_gates[0].matrix
            for i in range(1, len(qu_gates)):
                matrix = np.kron(matrix, qu_gates[i].matrix)

            return QuGate(matrix)
        else:
            raise qudot_errors.InvalidQuGateError("No gates specified")


class QuCircuit(object):
    """Representation of a quantum circuit and methods to run it.

    QuCircuit represents a quantum circuit with a list of tuples that
    represent operations (ops). The tuples have the form (QuGate, bit_list).
    QuGate is the gate operation you
    want performed and bit_list is the index of bits (starting from 1) that
    you want the operation to be performed. If you specify 'None' as the
    bit_list then the associated QuGate will be applied to the entire state.
    Before running or stepping through a circuit you must set the input state
    in the in_qu_state attribute or an error will be thrown

    Attributes:
        in_qu_state: The input QuState that you want to run the circuit on

        step_op_index: If you are stepping through a circuit, this tell you
                       your current index in the ops list
    """

    @property
    def in_qu_state(self):
        """Getter for the input QuState"""
        return self._in_qu_state

    @in_qu_state.setter
    def in_qu_state(self, qu_state_value):
        """Setter for the input QuState

        Will raise a QuCircuitError if you try to set the input QuState while
        stepping through a circuit
        """
        if self._step_op_index > 0:
            message = ("Trying to reset input QuState while stepping through"
                      " a circuit! You should reset_circuit() first.")
            raise qudot_errors.QuCircuitError(message)

        self._in_qu_state = qu_state_value

    @property
    def step_op_index(self):
        """Get which op you are on if stepping through a circuit. 0 based"""
        return self._step_op_index

    def __init__(self, ops):
        """Initialize

        Args:
            ops: a list of tuples. Each tuple represents on operation on
                 the circuit. The first element of the tuple is the QuGate
                 you want applied. The second element is a list of bits
                 you want the gate applied to. If the bit_list is None then
                 the QuGate ill be applied to all bits.
        """
        if not ops:
            message = ("You must specify a list of tuples to define a"
                      " quantum circuit: [(QuGate, bit_list)]")
            raise qudot_errors.QuCircuitError(message)

        self.ops = ops
        self._step_op_index = 0
        self._in_qu_state = None

    def run_circuit(self):
        """Runs a complete cycle of the circuit

        Make sure in_qu_state is set or this will raise an error

        Returns:
            The result of the input state after applying the circuit

        Raises:
            QuCircuitError: if no in_qu_state is set
        """

        if self._in_qu_state:
            for op in self.ops:
                self._in_qu_state.apply_gate(op[0], op[1])

            return self._in_qu_state
        else:
            message = "An input QuState must be set to apply a circuit"
            raise qudot_errors.QuCircuitError(message)

    def step_circuit(self):
        """Step through operations one at a time.

        Make sure in_qu_state is set or this will raise an error.
        Operation counting starts at 0

        Returns:
            int: The next index in the ops list or 0 if end of list

        Raises:
            QuCircuitError: if no in_qu_state is set
        """
        if self._in_qu_state:
            op = self.ops[self._step_op_index]
            self._in_qu_state.apply_gate(op[0], op[1])
            self._step_op_index += 1
            if self._step_op_index == len(self.ops):
                self.reset_circuit()

            return self._step_op_index
        else:
            message = ("An input QuState must be set to step through a circuit"
                      " Set self.in_qu_state with the QuState you want to"
                      " step the circuit through.")
            raise qudot_errors.QuCircuitError(message)

    def reset_circuit(self):
        """Sets the step index back to 0"""
        self._step_op_index = 0


#######################################################################


def measurement_probability(amplitude):
    """Calculate probability of seeing a measurement

    Args:
        amplitude: the probability amplitude of a measurement

    Returns:
        A real between 0 and 1 number that represents the probability
        associated with the given amplitude
    """
    if amplitude:
        return (np.conjugate(amplitude) * amplitude).real
    else:
        return 0


def apply_gate(qu_gate, base_state):
    """Apply a gate to a quantum state.

    This will not alter the supplied quantum state, it will return a
    new QuState

    Args:
        qu_gate: The quantum gate you wish to apply
        base_state: The quantum state you wish to apply the gate to

    Returns:
        A new QuState object that represents the result of applying
        qu_gate to qu_state
    """
    result = np.asarray(qu_gate.matrix * base_state.state)
    return QuState.init_from_vector(result)

def tensor_gates(left_gate, right_gate):
    """Return the tensor product of two QuGates """
    new_matrix = np.kron(left_gate.matrix, right_gate.matrix)
    return QuGate(new_matrix)

def tensor_states(left_state, right_state):
    """Return the tensor product of two quantum states """
    new_state = np.kron(left_state.state, right_state.state)
    return QuState.init_from_vector(new_state)


#######################################################################
# Module constants ####################################################
#######################################################################

# constantly used (see what I did there?) qubits
ZERO = QuBit(QuBit.ZERO)
ONE = QuBit(QuBit.ONE)
PLUS = QuBit(QuBit.PLUS)
MINUS = QuBit(QuBit.MINUS)
QUBIT_MAP = {QuBit.ZERO: ZERO, QuBit.ONE: ONE,
             QuBit.PLUS: PLUS, QuBit.MINUS: MINUS}

# if you have ever done quantum computing you know why I did this
ROOT2 = 1/math.sqrt(2)

# simple/common gates
X = QuGate.init_from_str('0 1; 1 0')
Y = QuGate.init_from_str('0 -1j; 1j 0')
Z = QuGate.init_from_str('1 0; 0 -1')
H = QuGate.init_from_str('1 1; 1 -1', ROOT2)
CNOT = QuGate.init_from_str('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0')

__author__ = 'psakkaris'
