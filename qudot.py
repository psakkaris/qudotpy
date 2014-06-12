__author__ = 'psakkaris'

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

                self._hilbert_dimension = int(math.log(self._state.size, 2))
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
        """Create a QuState from a vector.

        This method will create a QuState given a raw vector as input.
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
                    self._hilbert_dimension)
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
                    self._hilbert_dimension)
                states_map[dirac_str] = probablility
            index += 1

        if qubit_index >= 0:
            if qubit_index > self._hilbert_dimension:
                raise ValueError("qubit_index=%s is out of range" %
                                 str(qubit_index))
            qubit_map = {}
            for dirac_state in states_map:
                state_index = qudot_utils.dirac_str_to_int(dirac_state)
                amplitude = self._state[state_index][0]
                qubit = dirac_state[qubit_index]
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

    def apply_gate(self, qu_gate):
        raise NotImplementedError


class QuGate(object):
    """A general quantum gate.

    QuGate is a representation of a quantum gate using numpy matrices. We
    check that the gate is unitary upon initialization

    Attributes:
        matrix: the matrix representation of the gate
        dagger: the hermitian (dagger) of the gate
    """
    @property
    def matrix(self):
        return self._matrix

    @property
    def dagger(self):
        return self._matrix.H

    def __init__(self, matrix_str, multiplier=-1):
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
        self._matrix = np.matrix(matrix_str, dtype='complex')
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

# simple/common gates
X = QuGate('0 1; 1 0')
Y = QuGate('0 1j; -1j 0')
Z = QuGate('1 0; 0 -1')
H = QuGate('1 1; 1 -1', 1/math.sqrt(2))