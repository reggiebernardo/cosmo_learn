import numpy as np
from matplotlib import pyplot as plt
from cosmo_learn.cosmo_learn import *
import time  # Import the time module


if __name__ == '__main__':
    start_time = time.time()  # start the timer

    # s8=0.834+/-0.016 (Planck); s8=0.745+/-0.039 (KiDS-450); see https://arxiv.org/abs/2008.11285
    # H0, Om0, w0, s8 = 70, 0.3, -1, 0.8

    # desi1 Flat wCDM https://arxiv.org/pdf/2404.03002 + planck s8
    H0, Om0, w0, s8 = 67.74, 0.3095, -0.997, 0.834

    my_cosmo_learn=CosmoLearn([H0, Om0, w0, s8], seed=14000605)
    mock_keys=['CosmicChronometers', 'SuperNovae', \
               'BrightSirens', 'RedshiftSpaceDistorsions']

    my_cosmo_learn.make_mock(mock_keys=mock_keys)
    my_cosmo_learn.train_gp()
    my_cosmo_learn.train_brr()

    # plots results
    fig, ax=my_cosmo_learn.show_mocks(show_input=False)
    my_cosmo_learn.show_trained_ml(ax=ax, method='GP', label='GP', \
                                   hatch='\\', rasterized=True)
    my_cosmo_learn.show_trained_ml(ax=ax, method='BRR', label='BRR', \
                                   hatch='|', rasterized=True, color='blue', alpha=0.15)

    [ax[i].grid(True, alpha=0.25) for i in range(len(my_cosmo_learn.mock_data.keys()))]
    ax[0].legend(loc='upper left', prop={'size': 9})
    fig.tight_layout(); fig.subplots_adjust(hspace=0.18)
    fig.savefig('figs/gp_brr.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    end_time = time.time()  # dnd the timer
    print(f"Execution time: {end_time - start_time:.2f} seconds")  # print the elapsed time