import numpy as np
from matplotlib import pyplot as plt
from cosmo_learn.cosmo_learn import *


if __name__ == "__main__":

    # 0 generate data through CosmoLearn class
    # desi1 Flat wCDM https://arxiv.org/pdf/2404.03002 + planck s8
    H0, Om0, w0, s8 = 67.74, 0.3095, -0.997, 0.834
    my_cosmo_learn=CosmoLearn([H0, Om0, w0, s8], seed=14000605)
    mock_keys=['CosmicChronometers', 'SuperNovae', 'BrightSirens', 'RedshiftSpaceDistorsions']
    my_cosmo_learn.make_mock(mock_keys=mock_keys)

    # 1 TRAINING
    # gp and brr
    my_cosmo_learn.train_gp()
    my_cosmo_learn.train_brr()

    # ann
    my_cosmo_learn.init_ann()
    my_cosmo_learn.train_ann(verbose=0)

    # mcmc
    prior_dict={'H0_min': 0, 'H0_max': 100, 'Om0_min': 0, 'Om0_max': 1, \
                'w0_min': -10, 'w0_max': 10, 's8_min': 0.2, 's8_max': 1.5}
    rd_fid_prior={'mu': 147.46, 'sigma': 0.28}
    llprob=lambda x: my_cosmo_learn.llprob_wcdm(x, prior_dict=prior_dict, rd_fid_prior=rd_fid_prior)
    p0=[70, 0.3, -1, 0.8, 147]
    nwalkers=15; dres=[0.05, 0.005, 0.01, 0.01, 0.005]; nburn=100; nmcmc=2000
    my_cosmo_learn.get_mcmc_samples(nwalkers, dres, llprob, p0, nburn=nburn, nmcmc=nmcmc)

    # ga-fisher
    fitness_func=lambda x: -2*llprob(x)
    prior_ga=[[prior_dict['H0_min'], prior_dict['H0_max']], [prior_dict['Om0_min'], prior_dict['Om0_max']], \
              [prior_dict['w0_min'], prior_dict['w0_max']], [prior_dict['s8_min'], prior_dict['s8_max']], \
              [rd_fid_prior['mu']-20*rd_fid_prior['sigma'], rd_fid_prior['mu']+20*rd_fid_prior['sigma']]]
    my_cosmo_learn.get_gaFisher_samples(fitness_func, prior_ga, \
                                        llprob=llprob, nsamples=(nmcmc-nburn)*nwalkers)


    # 2 SHOW RESULTS
    # gp and brr reconstruction
    fig, ax=my_cosmo_learn.show_mocks(show_input=False)
    my_cosmo_learn.show_trained_ml(ax=ax, method='GP', label='GP')
    my_cosmo_learn.show_trained_ml(ax=ax, method='BRR', color='blue', alpha=0.15, hatch='|', label='BRR')
    [ax[i].grid(True, alpha=0.25) for i in range(len(my_cosmo_learn.mock_data.keys()))]
    ax[0].legend(loc='upper left', prop={'size': 9})
    fig.tight_layout(); fig.subplots_adjust(hspace=0.18)
    plt.show()

    # ann loss
    fig, ax=my_cosmo_learn.show_ann_loss()
    fig.tight_layout()
    plt.show()

    # ann reconstruction
    fig, ax=my_cosmo_learn.show_mocks(show_input=False)
    my_cosmo_learn.show_trained_ml(ax=ax, method='ANN', color='darkgreen', alpha=0.15, hatch='x', label='ANN')
    [ax[i].grid(True, alpha=0.25) for i in range(len(my_cosmo_learn.mock_data.keys()))]
    ax[0].legend(loc='upper left', prop={'size': 9})
    fig.tight_layout(); fig.subplots_adjust(hspace=0.18)
    plt.show()

    # mcmc and ga-fisher reconstruction
    fig, ax=my_cosmo_learn.show_mocks(show_input=False)
    my_cosmo_learn.show_bestfit_curve(ax=ax, method='MCMC', label='MCMC', color='pink')
    my_cosmo_learn.show_bestfit_curve(ax=ax, method='GAFisher', color='orange', alpha=0.15, hatch='|', label='GA-Fisher')
    [ax[i].grid(True, alpha=0.25) for i in range(len(my_cosmo_learn.mock_data.keys()))]
    ax[0].legend(loc='upper left', prop={'size': 9})
    fig.tight_layout(); fig.subplots_adjust(hspace=0.18)
    plt.show()

    # mcmc and ga-fisher contours
    corner_plot = my_cosmo_learn.show_param_posterior(method='MCMC')
    my_cosmo_learn.show_param_posterior(method='GAFisher', fig=corner_plot, color='blue', show_truth=True)
    corner_plot.legend(['MCMC', 'GA-Fisher'], loc='upper right')
    corner_plot.tight_layout(); corner_plot.subplots_adjust(hspace=0, wspace=0)
    plt.show()
