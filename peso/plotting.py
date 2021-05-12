""" Plotting helpers """

import matplotlib.pylab as plt
from peso.io import to_spykes_population


def spykes_plot_population(pop=None, mean_psth=None, **kwargs):
    """

    """
    if pop is None or mean_psth is None:
        pop, mean_psth = to_spykes_population(**kwargs)

    # plot heatmap of average psth
    _ = plt.figure(figsize=(10, 10))
    pop.plot_heat_map(mean_psth, sortby=None, sortorder='ascend', normalize=None, colors=['viridis'])
    # or latency

    # Population PSTH
    plt.figure()
    pop.plot_population_psth(all_psth=mean_psth, event_name='Event',
                             colors=([.5, .5, .5], [0, .6, 0]))
