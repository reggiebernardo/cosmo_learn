import numpy as np
from matplotlib import pyplot as plt
from cosmo_learn.cosmo_learn import *
import time  # Import the time module


def plot_hists_mcmc(sim_indices=range(1), savefig=False, fname='hists_mcmc'):
    start_time = time.time()  # start the timer

    # s8=0.834+/-0.016 (Planck); s8=0.745+/-0.039 (KiDS-450); see https://arxiv.org/abs/2008.11285
    # H0, Om0, w0, s8 = 70, 0.3, -1, 0.8

    # desi1 Flat wCDM https://arxiv.org/pdf/2404.03002 + planck s8
    H0, Om0, w0, s8 = 67.74, 0.3095, -0.997, 0.834

    # pick data sets
    mock_keys=['CosmicChronometers', 'SuperNovae', 'BaryonAcousticOscillations', \
               'BrightSirens', 'RedshiftSpaceDistorsions']

    # prior values
    p0=[70, 0.3, -1, 0.8, 147]
    prior_dict={'H0_min': 0, 'H0_max': 100, 'Om0_min': 0, 'Om0_max': 1, \
                'w0_min': -10, 'w0_max': 10, 's8_min': 0.2, 's8_max': 1.5}
    rd_fid_prior={'mu': 147.46, 'sigma': 0.28}

    # MCMC parameters
    nwalkers=15; dres=[0.05, 0.005, 0.01, 0.01, 0.005]; nburn=100; nmcmc=4000

    fig, ax=plt.subplots(figsize=(10,10), nrows=5)

    # reference universe
    my_cosmo_learn=CosmoLearn([H0, Om0, w0, s8], seed=14000605)
    my_cosmo_learn.make_mock(mock_keys=mock_keys)

    llprob=lambda x: my_cosmo_learn.llprob_wcdm(x, prior_dict=prior_dict, rd_fid_prior=rd_fid_prior)
    my_cosmo_learn.get_mcmc_samples(nwalkers, dres, llprob, p0, nburn=nburn, nmcmc=nmcmc)

    H0_samples_rs=my_cosmo_learn.mcmc_samples[:, 0]
    Om0_samples_rs=my_cosmo_learn.mcmc_samples[:, 1]
    w0_samples_rs=my_cosmo_learn.mcmc_samples[:, 2]
    s8_samples_rs=my_cosmo_learn.mcmc_samples[:, 3]
    rd_samples_rs=my_cosmo_learn.mcmc_samples[:, 4]

    # initialize stacked samples
    all_H0, all_Om0, all_w0, all_s8, all_rd = [], [], [], [], []

    color_rep='red'
    ax[0].hist(H0_samples_rs, bins=20, density=True, alpha=0.7, histtype='step', color=color_rep, lw=3)
    ax[1].hist(Om0_samples_rs, bins=20, density=True, alpha=0.7, histtype='step', color=color_rep, lw=3)
    ax[2].hist(w0_samples_rs, bins=20, density=True, alpha=0.7, histtype='step', color=color_rep, lw=3)
    ax[3].hist(s8_samples_rs, bins=20, density=True, alpha=0.7, histtype='step', color=color_rep, lw=3)
    ax[4].hist(rd_samples_rs, bins=20, density=True, alpha=0.7, histtype='step', color=color_rep, lw=3)

    # loop over universes
    for rs in sim_indices:
        print(f"Running simulation {rs}...")
        my_cosmo_learn=CosmoLearn([H0, Om0, w0, s8], seed=rs)
        my_cosmo_learn.make_mock(mock_keys=mock_keys)

        llprob=lambda x: my_cosmo_learn.llprob_wcdm(x, prior_dict=prior_dict, rd_fid_prior=rd_fid_prior)
        my_cosmo_learn.get_mcmc_samples(nwalkers, dres, llprob, p0, nburn=nburn, nmcmc=nmcmc)

        H0_samples_rs=my_cosmo_learn.mcmc_samples[:, 0]
        Om0_samples_rs=my_cosmo_learn.mcmc_samples[:, 1]
        w0_samples_rs=my_cosmo_learn.mcmc_samples[:, 2]
        s8_samples_rs=my_cosmo_learn.mcmc_samples[:, 3]
        rd_samples_rs=my_cosmo_learn.mcmc_samples[:, 4]

        # append to the global lists
        all_H0.append(H0_samples_rs)
        all_Om0.append(Om0_samples_rs)
        all_w0.append(w0_samples_rs)
        all_s8.append(s8_samples_rs)
        all_rd.append(rd_samples_rs)

        color_ens='blue'
        ax[0].hist(H0_samples_rs, bins=20, density=True, alpha=0.1, histtype='step', color=color_ens, lw=2)
        ax[1].hist(Om0_samples_rs, bins=20, density=True, alpha=0.1, histtype='step', color=color_ens, lw=2)
        ax[2].hist(w0_samples_rs, bins=20, density=True, alpha=0.1, histtype='step', color=color_ens, lw=2)
        ax[3].hist(s8_samples_rs, bins=20, density=True, alpha=0.1, histtype='step', color=color_ens, lw=2)
        ax[4].hist(rd_samples_rs, bins=20, density=True, alpha=0.1, histtype='step', color=color_ens, lw=2)

    # Concatenate all samples for stacked posterior
    H0_all = np.concatenate(all_H0)
    Om0_all = np.concatenate(all_Om0)
    w0_all  = np.concatenate(all_w0)
    s8_all  = np.concatenate(all_s8)
    rd_all  = np.concatenate(all_rd)

    # plot stacked posteriors (in black, bold)
    ax[0].hist(H0_all, bins=20, density=True, alpha=0.8, histtype='step', color='green', lw=5)
    ax[1].hist(Om0_all, bins=20, density=True, alpha=0.8, histtype='step', color='green', lw=5)
    ax[2].hist(w0_all,  bins=20, density=True, alpha=0.8, histtype='step', color='green', lw=5)
    ax[3].hist(s8_all,  bins=20, density=True, alpha=0.8, histtype='step', color='green', lw=5)
    ax[4].hist(rd_all,  bins=20, density=True, alpha=0.8, histtype='step', color='green', lw=5)

    ax[0].axvline(x=H0, ls='--', color='black', lw=2, alpha=0.5)
    ax[1].axvline(x=Om0, ls='--', color='black', lw=2, alpha=0.5)
    ax[2].axvline(x=w0, ls='--', color='black', lw=2, alpha=0.5)
    ax[3].axvline(x=s8, ls='--', color='black', lw=2, alpha=0.5)
    ax[4].axvline(x=rd_fid_prior['mu'], ls='--', color='black', lw=2, alpha=0.5)

    ax[0].set_xlabel(r'$H_0$ [km/s/Mpc]'); ax[0].set_ylabel(r'$P(H_0)$')
    ax[1].set_xlabel(r'$\Omega_m$'); ax[1].set_ylabel(r'$P(\Omega_{m0})$')
    ax[2].set_xlabel(r'$w$'); ax[2].set_ylabel(r'$P(w)$')
    ax[3].set_xlabel(r'$S_8$'); ax[3].set_ylabel(r'$P(S_8)$')
    ax[4].set_xlabel(r'$r_{\rm D}$ [Mpc]'); ax[4].set_ylabel(r'$P(r_{\rm D})$')

    fig.subplots_adjust(hspace=0.4)
    if savefig:
        fig.savefig('figs/' + fname + '.pdf', bbox_inches='tight', dpi=300)

    end_time = time.time()  # dnd the timer
    print(f"Execution time: {end_time - start_time:.2f} seconds")  # print the elapsed time


if __name__ == '__main__':
    n_sims=100
    plot_hists_mcmc(sim_indices=np.arange(0, n_sims, 1), savefig=True, fname=f'hists_mcmc_{n_sims}')

