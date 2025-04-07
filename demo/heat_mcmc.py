import numpy as np
import matplotlib.pyplot as plt
from cosmo_learn.cosmo_learn import *
import time

def plot_heatmap_mcmc(sim_indices=np.arange(100), savefig=True, fname='heatmap_mcmc'):
    start_time = time.time()

    # Define bin ranges for each parameter
    bin_settings = {
        'H0': {'range': (66, 70), 'bins': 40},
        'Om0': {'range': (0.2, 0.4), 'bins': 40},
        'w0': {'range': (-1.3, -0.7), 'bins': 40},
        's8': {'range': (0.75, 0.9), 'bins': 40},
        'rd': {'range': (146.5, 148.5), 'bins': 40}
    }

    param_names = [r'$H_0$ [km s$^{-1}$Mpc$^{-1}$]', r'$\Omega_{m0}$', \
                   r'$w$', r'$\sigma_8$', r'$r_{\rm D}$ [Mpc]']
    param_indices = [0, 1, 2, 3, 4]
    true_vals = [67.74, 0.3095, -0.997, 0.834, 147.46]

    n_params = len(param_indices)
    all_heatmaps = [[] for _ in range(n_params)]
    bin_edges_all = []

    for rs in sim_indices:
        print(f"Running simulation {rs}...")
        cosmo = CosmoLearn([67.74, 0.3095, -0.997, 0.834], seed=rs)
        cosmo.make_mock(mock_keys=[
            'CosmicChronometers', 'SuperNovae', 'BaryonAcousticOscillations',
            'BrightSirens', 'RedshiftSpaceDistorsions'
        ])

        llprob = lambda x: cosmo.llprob_wcdm(
            x,
            prior_dict={
                'H0_min': 0, 'H0_max': 100,
                'Om0_min': 0, 'Om0_max': 1,
                'w0_min': -10, 'w0_max': 10,
                's8_min': 0.2, 's8_max': 1.5
            },
            rd_fid_prior={'mu': 147.46, 'sigma': 0.28}
        )

        cosmo.get_mcmc_samples(
            nwalkers=15,
            dres=[0.05, 0.005, 0.01, 0.01, 0.005],
            llprob=llprob,
            p0=[70, 0.3, -1, 0.8, 147],
            nburn=100,
            nmcmc=2000
        )

        samples = cosmo.mcmc_samples  # shape: (n_samples, 5)

        # Loop through parameters and accumulate histograms
        for i, param_index in enumerate(param_indices):
            param_samples = samples[:, param_index]
            bins = bin_settings[list(bin_settings.keys())[i]]['bins']
            range_ = bin_settings[list(bin_settings.keys())[i]]['range']

            hist, bin_edges = np.histogram(param_samples, bins=bins, range=range_, density=True)
            all_heatmaps[i].append(hist)

            if rs == sim_indices[0]:
                bin_edges_all.append(bin_edges)

    # Plot all heatmaps in a single figure
    fig, axs = plt.subplots(nrows=n_params, figsize=(10, 2.2 * n_params), sharex=True)

    for i in range(n_params):
        heat = np.array(all_heatmaps[i]).T  # shape (bins, simulations)
        bins = bin_edges_all[i]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        im = axs[i].imshow(
            heat,
            aspect='auto',
            origin='lower',
            extent=[sim_indices[0], sim_indices[-1] + 1, bin_centers[0], bin_centers[-1]],
            cmap='Reds'
            # cmap='viridis'
        )
        axs[i].axhline(true_vals[i], color='b', linestyle='--', lw=1.5, alpha=0.7)
        axs[i].set_ylabel(param_names[i], fontsize=12)

    axs[-1].set_xlabel('Realization Index', fontsize=12)
    # fig.colorbar(im, ax=axs, label='Density', shrink=0.7)
    # plt.suptitle('MCMC Posterior Density Evolution Across Realizations', fontsize=14)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=0)

    if savefig:
        fig.savefig(f'figs/{fname}.pdf', bbox_inches='tight', dpi=300)

    print(f"Execution time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    plot_heatmap_mcmc(sim_indices=np.arange(0, 100, 1), savefig=True, fname='heatmap_mcmc')
