import numpy as np
from peso import peso


def identify_events(ts, bin_size=.05, number_of_neurons_treshold=20, minimum_time_between_states=0.15):
    """
    Find spontaneous periods of quiecence in spiketimes
    :param ts:
    :param bin_size:
    :param number_of_neurons_treshold:
    :param minimum_time_between_states:
    :return:
    """
    lfp = peso.bin_neuron(np.sort(ts), bin_size=bin_size)
    down_states = np.where(lfp < number_of_neurons_treshold)[0]
    down_states_lengths = np.diff(down_states)
    print(f'Eliminating {down_states[1:][down_states_lengths < .15 / bin_size].shape} that are too short')
    down_states = down_states[1:][down_states_lengths > minimum_time_between_states / bin_size]
    print(f'Ended up with {down_states.shape} down states')
    # convert into seconds:
    down_states = down_states * bin_size
    down_states -= .03
    print(down_states[(down_states > 63) & (down_states < 68)])  # compare with raster
    return lfp, down_states
