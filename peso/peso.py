"""Main module."""

import pandas as pd
import numpy as np
from tqdm import tqdm
import xarray as xr


def bin_neuron(spike_times, bin_size=.100, window=None):
    """
    Make binned raster for a single neuron
    :param spike_times:
    :param bin_size: in sec
    :param window:
    :return:
    """
    if window is None:
        window = [0, spike_times.max()]
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    return np.histogram(spike_times, bins)[0]


def bin_neurons(spike_times, ids, bin_size=None, window=None, plotose=False):
    """
    Make binned raster for many neurons
    :param spike_times:
    :param ids:
    :param bin_size: in sec
    :param window:
    :param plotose:
    :return:
    """
    if window is None:
        window = [0, spike_times.max()]
    # the following uses an list comprehension for loop (look it up):
    spike_counts = [bin_neuron(spike_times[ids == Neuron_ID], bin_size, window)
                    for Neuron_ID in tqdm(np.unique(ids))]
    spike_counts = np.vstack(spike_counts)
    raster = xr.DataArray(spike_counts, coords=dict(Time=np.arange(window[0], window[1] + bin_size, bin_size)[:-1],
                                                    Neuron_ID=range(len(np.unique(ids)))),
                          dims=['Neuron_ID', 'Time'])
    if plotose:
        raster.plot(robust=True)
    return raster
