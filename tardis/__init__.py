from __future__ import absolute_import


from .tardis import OceanModelCube, load_phenomena
from . import coords
from . import slices
from . import utils

__all__ = ['OceanModelCube',
           'load_phenomena',
           'coords',
           'slices',
           'utils']

__version__ = '0.3.0'
