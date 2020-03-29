# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: enable=missing-docstring

# import...
# ...from HydPy
from hydpy.core import sequencetools


class K(sequencetools.AideSequence):
    """Float to help nhru iterating for Pegasus"""
    NDIM, NUMERIC = 0, False


class SN_Ratio(sequencetools.AideSequence):
    """Ratio of frozen precipitation to total precipitation [-]."""
    NDIM, NUMERIC = 1, False


class TempS(sequencetools.AideSequence):
    """Temperatur der Schneedecke (temperature of the snow layer) [°C].

    Note that the value of sequence |TempS| is |nan| for snow-free surfaces.
    """
    NDIM, NUMERIC = 1, False
