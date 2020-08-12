from collections import namedtuple

from .base import WaveFunction
from .paulinet import PauliNet

__all__ = ['WaveFunction', 'PauliNet']

AnsatzSpec = namedtuple('AnsatzSpec', 'name entry defaults uses_workdir')

ANSATZES = [AnsatzSpec('paulinet', PauliNet.from_hf, PauliNet.DEFAULTS(), True)]
ANSATZES = {spec.name: spec for spec in ANSATZES}
