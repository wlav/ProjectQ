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
        import noise_traits
    except ImportError:
        return

    import logging
    log = logging.getLogger('ProjectQ')
    log.info('injecting noise ... ')

    gdict = globals()
    for gate in ['Rx', 'Ry', 'Rz', 'H', 'CNOT']:
       pdf, epsilon, args = None, 0, ()
       if hasattr(noise_traits, gate+'_pdf'):
          pdf = getattr(noise_traits, gate+'_pdf')
       if hasattr(noise_traits, gate+'_epsilon'):
          epsilon = getattr(noise_traits, gate+'_epsilon')
       if hasattr(noise_traits, gate+'_pdf_args'):
          args = getattr(noise_traits, gate+'_pdf_args')

       if not pdf and not epsilon:
          log.debug('no noise for gate %s', gate)
          continue

       log.debug('added noise to gate %s', gate)
       gdict[gate] = inject_noise(gdict[gate], pdf, epsilon, *args)

_enable_noise()
