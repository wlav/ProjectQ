#   Copyright 2017 ProjectQ-Framework (www.projectq.ch)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ._basics import (NotMergeable,
                      NotInvertible,
                      BasicGate,
                      SelfInverseGate,
                      BasicRotationGate,
                      ClassicalInstructionGate,
                      FastForwardingGate,
                      BasicMathGate,
                      BasicPhaseGate)
from ._command import apply_command, Command
from ._metagates import (DaggeredGate,
                         get_inverse,
                         ControlledGate,
                         C,
                         Tensor,
                         All)
from ._gates import *
from ._qftgate import QFT, QFTGate
from ._qubit_operator import QubitOperator
from ._shortcuts import *
from ._time_evolution import TimeEvolution

from ._noise import inject_noise

def _enable_noise():
    try:
        import noise_model
    except ImportError:
        return

    import logging
    log = logging.getLogger('ProjectQ')
    log.info('injecting noise models ... ')

    gdict = globals()
    for gate in ['Rx', 'Ry', 'Rz', 'H', 'CNOT']:
       try:
          gdict[gate] = inject_noise(
             gdict[gate],
             getattr(noise_model, gate+'_model'),
             *getattr(noise_model, gate+'_args'))
          log.debug('added noise model for %s', gate)
       except AttributeError, e:
          log.debug('no noise model for %s (%s)', gate, str(e))
          pass

_enable_noise()
