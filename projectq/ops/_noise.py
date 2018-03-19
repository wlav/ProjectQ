#   Copyright 2018 UC Regents
#   Licensed under LBNL BSD.

"""
Contains meta gates that add noise.
* LocalNoiseGate (Generic gate adding noise local to a specific gate)

As well as the create function
* inject_noise (Wrapps a gate with the specific noise creator for it)
"""

from ._basics import BasicGate
from ._metagates import get_inverse


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
        Return the inverse gate. Since noise was added, the inverse is not
        going to be exact (TODO: perhaps drop noise for inverse?).
        """

        return LocalNoiseGate(get_inverse(self._gate), self._dist, *self._args)

    def __or__(self, qubits):
        """
        Apply the gate with noise to qubits according to the sampling
        distribution given.

        Args:
            qubits (tuple of lists of Qubit objects): qubits to which to apply
                the gate.
        """

        qubits = BasicGate.make_tuple_of_qureg(qubits)
        for qb in qubits:
            self._apply_with_noise(self._gate, self._dist(*self._args), qb)

    def _apply_with_noise(self, gate, rnd, qb):
        gate | qb             # default: no noise

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

            Rx = NoisyAngleGate(Rx, lambda: random.gauss(0., 0.05))
            Rx | x

    will add stochastic noise using gaussian sampling to H.
    """

    def _apply_with_noise(self, gate, rnd, qb):
        noisy_gate = gate.__class__(gate.angle + rnd)
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

    if gate in (Rx, Ry, Rz):
        return NoisyAngleGateFactory(gate, distribution, *args)

    return LocalNoiseGate(gate, distribution, *args)
