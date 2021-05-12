""" input/output """

import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def from_phy(path_to_data='/Users/myroshnychenkm2/Downloads/dataset/', sampling_frequency=30000):
    """
    Get spikes from a kilosort/phy result folder
    :param path_to_data:
    :param sampling_frequency:
    :return:
    :id: neuron id, 1xN
    :ts: corresponding spiketime, 1xN
    """
    groupfname = os.path.join(path_to_data, 'cluster_groups.csv')
    groups = pd.read_csv(groupfname, delimiter='\t')

    # load spike times and cluster IDs
    with open(path_to_data + 'spike_clusters.npy', 'rb') as f:
        ids = np.load(f).flatten()
    with open(path_to_data + 'spike_times.npy', 'rb') as f:
        ts = np.load(f).flatten()

    # Create the list of our "good" labeled units
    ids_to_take = groups[(groups.group == 'good')].cluster_id
    # Find which spikes beloing to our "good" groups
    spikes_to_take = []
    for i in tqdm(ids_to_take, desc='Selecting only good spikes'):
        spikes_to_take.extend((ids == i).nonzero()[0])
    # only take spikes that are in our list
    ids = np.array(ids[spikes_to_take])
    ts = np.array(ts[spikes_to_take]).astype(float) / sampling_frequency

    return ids, ts



def events(trial_starts):
    """
    Simple wrapper creating a dataframe with times we want to lock onto
    :param trial_starts: List of times of interest (trials)
    :return: spykes object
    """
    return pd.DataFrame({'trialStart': trial_starts})


def to_spykes(s_ts, s_id, debug=False):
    """
    Use spykes library
    :param s_ts:
    :param s_id:
    :param debug:
    :return:
    """

    def print_spyke(spykess):
        [print(len(spykess[i].spiketimes)) for i in range(len(spykess))]

    from spykes.plot import neurovis
    s_id = s_id.astype('int')
    neuron_list = list()
    for iu in np.unique(s_id):
        spike_times = s_ts[s_id == iu]
        if len(spike_times) < 2:
            if debug:
                print('Too few spiketimes in this unit: ' + str(spike_times))
        else:
            neuron = neurovis.NeuroVis(spike_times, name='ram' + str(iu))
            neuron_list.append(neuron)

    if debug:
        print_spyke(neuron_list)
    return neuron_list


def to_spykes_population(spikes, events, event='trialStart', window=(-100, 100), bin_size=10, fr_thr=.1):
    """

    :param spikes: List of NeuroVis objects
    :param spykes_df:
    :param event:
    :param window:
    :param bin_size:
    :param fr_thr:
    :param plotose:
    :return:
    """
    import spykes
    assert window[1] - window[0] > 0, 'Window size must be greater than zero!'

    # filter firing rate
    spikes = [i for i in spikes if i.firingrate > fr_thr]
    pop = spykes.plot.popvis.PopVis(spikes)

    # calculate psth
    mean_psth = pop.get_all_psth(event=event, df=events, window=window, binsize=bin_size, plot=False)

    # check
    if mean_psth['data'][0].size == 0:
        raise IndexError('Empty group PSTH!')

    return pop, mean_psth
