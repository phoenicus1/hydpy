# -*- coding: utf-8 -*-
"""This module implements statistical functionalities frequently used in
hydrological modelling.
"""
# import...
# ...from standard library
from __future__ import division, print_function
# ...from site-packages
import numpy
import pandas
import wrapt
# ...from HydPy
from hydpy.core import autodoctools
from hydpy.core import objecttools
from hydpy.auxs import validtools


def errorhandler(description):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        try:
            return wrapped(*args, **kwargs)
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to %s' % description)
    return wrapper


@errorhandler('the weighted mean time')
def calc_mean_time(timepoints, weights):
    """Return the weighted mean of the given timepoints.

    With equal given weights, the result is simply the mean of the given
    time points:

    >>> from hydpy.auxs.statstools import calc_mean_time
    >>> calc_mean_time(timepoints=[3., 7.],
    ...                weights=[2., 2.])
    5.0

    With different weights, the resulting mean time is shifted to the larger
    ones:

    >>> calc_mean_time(timepoints=[3., 7.],
    ...                weights=[1., 3.])
    6.0

    Or, in the most extreme case:

    >>> calc_mean_time(timepoints=[3., 7.],
    ...                weights=[0., 4.])
    7.0

    There will be some checks for input plausibility perfomed, e.g.:

    >>> calc_mean_time(timepoints=[3., 7.],
    ...                weights=[-2., 2.])
    Traceback (most recent call last):
    ...
    ValueError: While trying to calculate the weighted mean time, \
the following error occured: For the following objects, at least \
one value is negative: weights.
    """
    timepoints = numpy.array(timepoints)
    weights = numpy.array(weights)
    validtools.test_equal_shape(timepoints=timepoints, weights=weights)
    validtools.test_non_negative(weights=weights)
    return numpy.dot(timepoints, weights)/numpy.sum(weights)


@errorhandler('the weighted time deviation from mean time')
def calc_mean_time_deviation(timepoints, weights, mean_time=None):
    """Return the weighted deviation of the given timepoints from their mean
    time.

    With equal given weights, the is simply the standard deviation of the
    given time points:

    >>> from hydpy.auxs.statstools import calc_mean_time_deviation
    >>> calc_mean_time_deviation(timepoints=[3., 7.],
    ...                          weights=[2., 2.])
    2.0

    One can pass a precalculated or alternate mean time:

    >>> from hydpy import round_
    >>> round_(calc_mean_time_deviation(timepoints=[3., 7.],
    ...                                 weights=[2., 2.],
    ...                                 mean_time=4.))
    2.236068

    >>> round_(calc_mean_time_deviation(timepoints=[3., 7.],
    ...                                 weights=[1., 3.]))
    1.732051

    Or, in the most extreme case:

    >>> calc_mean_time_deviation(timepoints=[3., 7.],
    ...                          weights=[0., 4.])
    0.0

    There will be some checks for input plausibility perfomed, e.g.:

    >>> calc_mean_time_deviation(timepoints=[3., 7.],
    ...                          weights=[-2., 2.])
    Traceback (most recent call last):
    ...
    ValueError: While trying to calculate the weighted time deviation \
from mean time, the following error occured: For the following objects, \
at least one value is negative: weights.
    """
    timepoints = numpy.array(timepoints)
    weights = numpy.array(weights)
    validtools.test_equal_shape(timepoints=timepoints, weights=weights)
    validtools.test_non_negative(weights=weights)
    if mean_time is None:
        mean_time = calc_mean_time(timepoints, weights)
    return (numpy.sqrt(numpy.dot(weights, (timepoints-mean_time)**2) /
                       numpy.sum(weights)))


def _prepare_values(sim, obs, node, skip_nan):
    if node:
        if sim is not None:
            raise ValueError(
                'do not pass sim and node')
        if obs is not None:
            raise ValueError(
                'do not pass obs and node')
        sim = node.sequences.sim.series
        obs = node.sequences.obs.series
    elif (sim is not None) and (obs is None):
        raise ValueError(
            'when sim must pass obs')
    elif (obs is not None) and (sim is None):
        raise ValueError(
            'when obs musts pass sim')
    sim = numpy.asarray(sim)
    obs = numpy.asarray(obs)
    if skip_nan:
        idxs = ~numpy.isnan(sim) * ~numpy.isnan(obs)
        sim = sim[idxs]
        obs = obs[idxs]
    return sim, obs


@errorhandler('calculate the Nash-Sutcliffe efficiency')
def nse(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the efficiency criteria after Nash & Sutcliffe."""
    sim, obs = _prepare_values(sim, obs, node, skip_nan)
    return 1.-numpy.sum((sim-obs)**2)/numpy.sum((obs-numpy.mean(obs))**2)


@errorhandler('calculate the absolute bias')
def bias_abs(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the absolute difference between the means of the simulated
    and the observed values."""
    sim, obs = _prepare_values(sim, obs, node, skip_nan)
    return numpy.mean(sim-obs)


@errorhandler('calculate the relative bias')
def bias_rel(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the relative difference between the means of the simulated
    and the observed values."""
    sim, obs = _prepare_values(sim, obs, node, skip_nan)
    return numpy.mean(sim)/numpy.mean(obs)-1.


@errorhandler('calculate the standard deviation ratio')
def std_ratio(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the ratio between the standard deviation of the simulated
    and the observed values."""
    sim, obs = _prepare_values(sim, obs, node, skip_nan)
    return numpy.var(sim)/numpy.var(obs)-1.


@errorhandler('calculate the Pearson correlation coefficient')
def corr(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the product-moment correlation coefficient after Pearson."""
    sim, obs = _prepare_values(sim, obs, node, skip_nan)
    return numpy.corrcoef(sim, obs)[0, 1]


@errorhandler('...')
def evaluate(nodes, criteria, node_names=None, criterion_names=None,
             skip_nan=False):
    if node_names:
        if len(nodes) != len(node_names):
            raise ValueError(
                'shape mismatch')
    else:
        node_names = [node.name for node in nodes]
    if criterion_names:
        if len(criteria) != len(criterion_names):
            raise ValueError(
                'shape mismatch')
    else:
        criterion_names = [criterion.__name__ for criterion in criteria]
    data = numpy.empty((len(nodes), len(criteria)), dtype=float)
    for idx, node in enumerate(nodes):
        sim, obs = _prepare_values(None, None, node, skip_nan)
        for jdx, criterion in enumerate(criteria):
            data[idx, jdx] = criterion(sim, obs)
    table = pandas.DataFrame(
        data=data, index=node_names, columns=criterion_names)
    return table




autodoctools.autodoc_module()
