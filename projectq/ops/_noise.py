#   Copyright 2018 UC Regents
#   Licensed under LBNL BSD.

"""
Contains meta gates that add noise.
* LocalNoiseGate (Generic gate adding noise local to a specific gate)

As well as the create function
* inject_noise (Wrapps a gate with the specific noise creator for it)
"""

import copy, random
import numpy as np

from ._basics import BasicGate, SelfInverseGate
from ._metagates import get_inverse


class IGate(SelfInverseGate):
    """ Identity gate class """
    def __str__(self):
        return "I"

    @property
    def matrix(self):
        return np.matrix([[1, 0], [0, 1]])

I = IGate()

class PartialXGate(SelfInverseGate):
    """ Partial bit-flip (X) gate class """
    def __str__(self):
        return "PX"

    @property
    def matrix(self):
        return np.matrix([[0, 1], [1, 0]])


class LocalNoiseGate(BasicGate):
    """
    Wrapper class adding localized noise to a gate.

    Example:
        .. code-block:: python

            H = LocalNoiseGate(H, lambda: random.gauss(0., 0.05))
            H | x

    will add stochastic noise using gaussian sampling to H.
    """

    def __init__(self, gate, distribution, *args):
        """
        Initialize a LocalNoiseGate representing the a noisy version of the
        gate 'gate'.

        Args:
            gate: Any gate object to which noise will be added.
            distribution: Callable object of the sampling distribution to use.
        """

        BasicGate.__init__(self)
        self._gate = gate
        self._dist = distribution
        self._args = args

    def update_model(self, distribution, *args):
        self._dist = distribution
        self._args = args

    def __str__(self):
        """
        Return string representation (str(gate) + \"_noisy\").
        """

        return str(self._gate) + "_noisy"

    def tex_str(self):
        """
        Return the Latex string representation of a Daggered gate.
        """

        if hasattr(self._gate, 'tex_str'):
            return self._gate.tex_str() + r"${}_{noisy}$"
        else:
            return str(self._gate) + r"${}_{noisy}$"

    def get_inverse(self):
        """
        Return the inverse gate. Since noise was added, the inverse does not
        have to be exact (TODO: perhaps drop noise for inverse?).
        """
        return LocalNoiseGate(get_inverse(self._gate), self._dist, *self._args)

    def __eq__(self, other):
        """
        Return True if both wrapper and wrapped gates are equal.
        """
        return isinstance(other, self.__class__) and self._gate == other._gate


class NoisyAngleGate(LocalNoiseGate):
    """
    Wrapper class adding stochastic noise to a gate by means of an
    additional rotation angle.

    Example:
        .. code-block:: python

            Rx = NoisyAngleGate(Rx, random.gauss, 0., 0.05)
            Rx | x

    will add stochastic noise using gaussian sampling to Rx.
    """

    def __or__(self, qubits):
        """
        Apply the gate with noise to qubits according to the sampling
        distribution given.

        Args:
            qubits (tuple of lists of Qubit objects): qubits to which to apply
                the gate.
        """

        gate = self._gate
        rnd_angle = self._dist(*self._args)
        qubits = BasicGate.make_tuple_of_qureg(qubits)
        for qb in qubits:
            noisy_gate = gate.__class__(gate.angle + rnd_angle)
            noisy_gate | qb


class NoisyAngleGateFactory(object):
    def __init__(self, gate_type, distribution, *args):
        self._type = gate_type
        self._dist = distribution
        self._args = args

    def update_model(self, distribution, *args):
        self._dist = distribution
        self._args = args

    def __call__(self, *args):
        return NoisyAngleGate(self._type(*args), self._dist, *self._args)

    def get_name(self):
        return self._type.__name__
    __name__ = property(get_name)


class NoisyCNOTGate(LocalNoiseGate):
    """
    Wrapper class adding stochastic noise to a CNOT through rotations on the
    target and control qubit, and random failure.

    Example:
        .. code-block:: python

            # distribution to return noise for target and control, respectively
            def gauss2(mu1, sigma1, mu2, sigma2):
                return random.gauss(mu1, sigma1), random.gauss(mu2, sigma2)

            CNOT = NoisyCNOTGate(gauss2, 0, 0.1*math.pi, 0, 0.1*math.pi)
            CNOT | x

    will add stochastic noise using gaussian sampling to CNOT.
    """

    def __init__(self, distribution, *args):
        from ._shortcuts import CNOT
        LocalNoiseGate.__init__(self, CNOT, distribution, *args)
        self.threshold = 1.0

    def __or__(self, qubits):
        """
        Apply the gate with noise to qubits according to the sampling
        distribution given.

        Args:
            qubits (tuple of lists of Qubit objects): qubits to which to apply
                the gate.
        """

        # TODO: this is only b/c CNOT is an object; would prefer to fit a
        # factory in somewhere (and anyway not to have to touch internals)

        noisy_gate = copy.deepcopy(self._gate)
        if random.random() > self.threshold:
            noisy_gate._gate = I

        return noisy_gate | qubits


def inject_noise(gate, distribution, *args):
    """
    Wrapper creator specific to the given gate to add stochastic noise
    that follows the sampling distribution.

    Example:
        .. code-block:: python

            Ry = make_noisy(Ry, random.gauss, 0., 0.05)
            Ry | psi

    will add stochastic noise using gaussian sampling to H.
    """

    from ._gates import H, Rx, Ry, Rz
    from ._shortcuts import CNOT

    if gate in (Rx, Ry, Rz):
        return NoisyAngleGateFactory(gate, distribution, *args)

    if gate in (CNOT,):
        return NoisyCNOTGate(distribution, *args)

    return gate
