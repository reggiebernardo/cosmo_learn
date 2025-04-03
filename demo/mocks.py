import numpy as np
from matplotlib import pyplot as plt
from cosmo_learn.cosmo_learn import *
import time  # Import the time module


def plot_mocks(sim_indices=range(10), savefig=False, fname='mocks'):
    start_time = time.time()  # start the timer

    # s8=0.834+/-0.016 (Planck); s8=0.745+/-0.039 (KiDS-450); see https://arxiv.org/abs/2008.11285
    # H0, Om0, w0, s8 = 70, 0.3, -1, 0.8

    # desi1 Flat wCDM https://arxiv.org/pdf/2404.03002 + planck s8
    H0, Om0, w0, s8 = 67.74, 0.3095, -0.997, 0.834

    my_cosmo_learn=CosmoLearn([H0, Om0, w0, s8], seed=14000605)
    mock_keys=['CosmicChronometers', 'SuperNovae', 'BaryonAcousticOscillations', \
               'BrightSirens', 'RedshiftSpaceDistorsions']
    my_cosmo_learn.make_mock(mock_keys=mock_keys)

    # fig, ax=my_cosmo_learn.show_mocks(show_input=True) # only mocks, no residuals
    fig, ax=my_cosmo_learn.show_mocks_and_residuals(show_input=True, fmt_train='r*', fmt_test='r*', \
                                                    alpha_all=0.7, alpha_sne=0.1)
    fig.subplots_adjust(wspace=0.05)

    # loop with changing seed
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for rs in sim_indices:
        my_cosmo_learn_rs=CosmoLearn([H0, Om0, w0, s8], seed=rs)
        my_cosmo_learn_rs.make_mock(mock_keys=mock_keys)
        # color = colors[rs % len(colors)]
        color='b'
        my_cosmo_learn_rs.show_mocks_and_residuals(ax=ax, fmt_train=f'{color}o', fmt_test=f'{color}o', \
                                                   alpha_all=0.03, alpha_sne=0.008)
        
    # plot marker cosmology again---adds emphasis to the plot
    my_cosmo_learn=CosmoLearn([H0, Om0, w0, s8], seed=14000605)
    my_cosmo_learn.make_mock(mock_keys=mock_keys)
    my_cosmo_learn.show_mocks_and_residuals(ax=ax, show_input=True, fmt_train='r*', fmt_test='r*', \
                                            alpha_all=0.7, alpha_sne=0.1)

    # ax.set_rasterized(True)
    # loop through each axis if ax is an array
    if isinstance(ax, np.ndarray):
        for axis in ax.flatten():
            axis.set_rasterized(True)
    else:
        ax.set_rasterized(True)

    if savefig:
        fig.savefig('figs/' + fname + '.pdf', bbox_inches='tight', dpi=300)

    end_time = time.time()  # dnd the timer
    print(f"Execution time: {end_time - start_time:.2f} seconds")  # print the elapsed time


if __name__ == '__main__':
    plot_mocks(sim_indices=np.arange(0, 100, 1), savefig=True, fname='mocks')

