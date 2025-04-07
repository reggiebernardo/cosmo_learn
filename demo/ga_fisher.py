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
    mock_keys=['CosmicChronometers', 'SuperNovae', 'BaryonAcousticOscillations', \
               'BrightSirens', 'RedshiftSpaceDistorsions']
    my_cosmo_learn.make_mock(mock_keys=mock_keys)
    
    # mcmc 
    prior_dict={'H0_min': 0, 'H0_max': 100, 'Om0_min': 0, 'Om0_max': 1, \
                'w0_min': -10, 'w0_max': 10, 's8_min': 0.2, 's8_max': 1.5}
    rd_fid_prior={'mu': 147.46, 'sigma': 0.28}
    llprob=lambda x: my_cosmo_learn.llprob_wcdm(x, prior_dict=prior_dict, rd_fid_prior=rd_fid_prior)

    p0=[70, 0.3, -1, 0.8, 147]
    nwalkers=15; dres=[0.05, 0.005, 0.01, 0.01, 0.005]; nburn=100; nmcmc=2000
    my_cosmo_learn.get_mcmc_samples(nwalkers, dres, llprob, p0, nburn=nburn, nmcmc=nmcmc);

    # ga-fisher
    fitness_func=lambda x: -2*llprob(x)
    prior_ga=[[prior_dict['H0_min'], prior_dict['H0_max']], [prior_dict['Om0_min'], prior_dict['Om0_max']], \
              [prior_dict['w0_min'], prior_dict['w0_max']], [prior_dict['s8_min'], prior_dict['s8_max']], \
              [rd_fid_prior['mu']-20*rd_fid_prior['sigma'], rd_fid_prior['mu']+20*rd_fid_prior['sigma']]]

    # change ga hyperparameters via my_cosmo_learn.ga_params[key]=new_values
    # some defaults: 'max_num_iteration'=1000, 'population_size'=100, 'mutation_probability'=0.3
    # my_cosmo_learn.ga_params['max_num_iteration']=100; my_cosmo_learn.ga_params['population_size']=50
    my_cosmo_learn.get_gaFisher_samples(fitness_func, prior_ga, \
                                        llprob=llprob, nsamples=(nmcmc-nburn)*nwalkers);


    # make best fit plot
    fig, ax=my_cosmo_learn.show_mocks(show_input=True)
    my_cosmo_learn.show_bestfit_curve(ax=ax, method='MCMC', label='MCMC', color='pink')
    my_cosmo_learn.show_bestfit_curve(ax=ax, method='GAFisher', color='orange', alpha=0.15, hatch='|', label='GA-Fisher')

    [ax[i].grid(True, alpha=0.25) for i in range(len(my_cosmo_learn.mock_data.keys()))]
    ax[0].legend(loc='upper left', prop={'size': 9})
    fig.tight_layout(); fig.subplots_adjust(hspace=0.18)
    fig.savefig('figs/best_fit_ga_fisher.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # make corner plot
    corner_plot = my_cosmo_learn.show_param_posterior(method='MCMC')
    my_cosmo_learn.show_param_posterior(method='GAFisher', fig=corner_plot, color='blue', ls='--', show_truth=True)

    corner_plot.legend(['MCMC', 'GA-Fisher'], loc='upper right')
    corner_plot.tight_layout(); corner_plot.subplots_adjust(hspace=0, wspace=0)
    plt.savefig('figs/corner_ga_fisher.pdf', bbox_inches='tight')
    plt.show()

    end_time = time.time()  # dnd the timer
    print(f"Execution time: {end_time - start_time:.2f} seconds")  # print the elapsed time

