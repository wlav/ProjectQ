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
from math import sin, cos

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


class LocalNoiseBase(BasicGate):
    """
    Helper to capture common structure of noisy gate implementations.
    """

    def __init__(self, pdf, frate, *args):
        """
        Initialize a LocalNoiseBase .

        Args:
            pdf:   callable object of the sampling distribution to use.
            frate: failure rate; chance of no-op instead of gate
            *args: argument to use when calling the pdf
        """
        BasicGate.__init__(self)
        self._pdf   = pdf
        self._frate = frate
        self._args  = args

    def update_model(self, pdf=None, frate=None, args=None):
        if pdf is not None:
            self._pdf = pdf
        if frate is not None:
            self._frate = frate
        if args is not None:
            self._args = args

    def noise(self):
        return self._pdf(*self._args)

    def failure(self):
        if self._frate and random.random() > 1.-self._frate:
            return True
        return False

    def __eq__(self, other):
        # BasicGate calls two gates equal if their matrices (if any) are
        # 'close', but if the matrix contains a small amount of noise,
        # they really aren't the same. For simplicity and to prevent the
        # optimizer to unequally reduce certain gates, only equate equal
        # to self.
        return self is other


class SelfInverseLocalNoiseBase(LocalNoiseBase):
    def get_inverse(self):
        return self.__class__(self._pdf, self._frate, *self._args)


class XNoiseGate(SelfInverseLocalNoiseBase):
    """ Random rotation in X due to noise gate class """

    def __str__(self):
        return "PX"

    @property
    def matrix(self):
        # effectively: Rx(self.noise())
        rnd = self.noise()*0.5
        return np.matrix([[    cos(rnd), -1j*sin(rnd)],
                          [-1j*sin(rnd),     cos(rnd)]])

class NoisyNOTGate(SelfInverseLocalNoiseBase):
    """NOT Gate + random rotation in X gate class"""

    def __str__(self):
        return "PNOT"

    @property
    def matrix(self):
        # effectively Ry(self.noise())*X
        rnd = self.noise()*0.5
        return np.matrix([[-1j*sin(rnd),     cos(rnd)],
                          [    cos(rnd), -1j*sin(rnd)]])


class XYNoiseGate(LocalNoiseBase):
    """ Random rotation in X+Y gate class """

    def __init__(self, ratio, pdf, frate, *args):
        """
        Initialize a XYNoiseGate .

        Args:
            ratio: relative strenght of noise direction X/Y
            pdf:   callable object of the sampling distribution to use.
            frate: failure rate; chance of no-op instead of gate
            *args: argument to use when calling the pdf
        """
        LocalNoiseBase.__init__(self, pdf, frate, *args)
        self._ratio = ratio
        self._xweight = 0.5*(ratio/(ratio+1.)) # 0.5 from convention
        self._yweight = 0.5*(   1./(ratio+1.)) # id.

    def __str__(self):
        return "WXY"

    def get_inverse(self):
        return YXNoiseGate(self._ratio, self._pdf, self._frate, *self._args)

    def _Xwobble(self, rnd):
        # split the wobble according to the ratio ...
        rnd *= self._xweight
        return np.matrix([[    cos(rnd), -1j*sin(rnd)],
                          [-1j*sin(rnd),     cos(rnd)]])

    def _Ywobble(self, rnd):
        # ... and likewise in Y
        rnd *= self._yweight
        return np.matrix([[cos(rnd), -sin(rnd)],
                          [sin(rnd),  cos(rnd)]])

    @property
    def matrix(self):
        # X applied first, then Y
        rnd = self.noise()
        return self._Ywobble(rnd)*self._Xwobble(rnd)

class YXNoiseGate(XYNoiseGate):
    """ Random rotation in Y+X gate class """

    def __str__(self):
        return "WYX"

    def get_inverse(self):
        return XYNoiseGate(self._ratio, self._pdf, self._frate, *self._args)

    @property
    def matrix(self):
        # Y applied first, then X
        rnd = self.noise()
        return self._Xwobble(rnd)*self._Ywobble(rnd)


class LocalNoiseGate(LocalNoiseBase):
    """
    Wrapper class adding localized noise to a gate.

    Example:
        .. code-block:: python

            H = LocalNoiseGate(H, lambda: random.gauss(0., 0.05))
            H | x

    will add stochastic noise using gaussian sampling to H.
    """

    def __init__(self, gate, pdf, frate, *args):
        """
        Initialize a LocalNoiseGate representing the a noisy version of the
        gate 'gate'.

        Args:
            gate: Any gate object to which noise will be added.
            pdf: Callable object of the sampling distribution to use.
            frate: failure rate; chance of no-op instead of gate
            *args: argument to use when calling the pdf
        """

        LocalNoiseBase.__init__(self, pdf, frate, *args)
        self._gate  = gate

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
        return LocalNoiseGate(get_inverse(self._gate), self._pdf, self._frate, *self._args)


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
        rnd_angle = self.noise()
        qubits = BasicGate.make_tuple_of_qureg(qubits)
        for qb in qubits:
            noisy_gate = gate.__class__(gate.angle + rnd_angle)
            noisy_gate | qb


class NoisyAngleGateFactory(LocalNoiseBase):
    """
    Angle gates are instances (e.g. Rx is a class, Rx(3.14) an instance), so
    forward the noise pdf, interject this factory as the "class."
    """

    def __init__(self, gate_type, pdf, frate, *args):
        LocalNoiseBase.__init__(self, pdf, frate, *args)
        self._type = gate_type

    def __call__(self, *args):
        return NoisyAngleGate(self._type(*args), self._pdf, self._frate, *self._args)

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

    def __init__(self, pdf, frate, *args):
        from ._shortcuts import CNOT
        LocalNoiseGate.__init__(self, CNOT, pdf, frate, *args)

        # CNOT is an object, so this is a global replacement of the gate
        # underlying this metagate
        def pdf1(*args):
            return pdf(*args)[1]
        self._gate._gate = NoisyNOTGate(pdf1, frate, *args)

        # gate to wobble the source
        def pdf0(*args):
            return pdf(*args)[0]
        self._control_noise = XNoiseGate(pdf0, frate, *args)

    def update_model(self, pdf=None, frate=None, args=None):
        # update both self as well as the underlying gates
        LocalNoiseGate.update_model(self, pdf, frate, args)
        self._control_noise.update_model(pdf, frate, args)
        self._gate._gate.update_model(pdf, frate, args)

    def __or__(self, qubits):
        """
        Apply the gate with noise to qubits according to the sampling
        distribution given.

        Args:
            qubits (tuple of lists of Qubit objects): qubits to which to apply
                the gate.
        """

        assert len(qubits) == 2
        if self.failure():
            # apply identity (i.e. total gate failure) to each qubit
            for qb in qubits:
                I | qb
        else:
            # apply noisy operation on target qubit
            self._gate | qubits

            # apply noise on control qubit
            self._control_noise | qubits[0]


def inject_noise(gate, pdf, frate, *args):
    """
    Wrapper creator specific to the given gate to add stochastic noise
    that follows the sampling distribution.

    Example:
        .. code-block:: python

            Ry = make_noisy(Ry, random.gauss, None, 0., 0.05)
            Ry | psi

    will add stochastic noise using gaussian sampling to H and no
    failure rate.
    """

    from ._gates import H, Rx, Ry, Rz
    from ._shortcuts import CNOT

    if gate in (Rx, Ry, Rz):
        return NoisyAngleGateFactory(gate, pdf, frate, *args)

    if gate in (CNOT,):
        return NoisyCNOTGate(pdf, frate, *args)

    return gate
