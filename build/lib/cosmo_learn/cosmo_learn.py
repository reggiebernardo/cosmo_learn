
import numpy as np
import matplotlib.pyplot as plt

from .LISA_bright import generate
from astropy.cosmology import w0waCDM
from scipy.special import hyp2f1
# from scipy.linalg import block_diag
from scipy.constants import c
c_kms=c/1000

from sklearn.model_selection import train_test_split

import time
from multiprocess import Pool, cpu_count
import emcee
from scipy.optimize import minimize

# from .ga_pygad import GeneticAlgorithm as GA
from geneticalgorithm import geneticalgorithm as GA
from numdifftools import Hessian

# from gp6.gp6 import GP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, RBF, WhiteKernel, \
                                              Matern, RationalQuadratic, \
                                              ExpSineSquared, DotProduct)

kernels_sk = {'RBF': ConstantKernel()*RBF()+WhiteKernel(), \
              'Matern': ConstantKernel()*Matern(), \
              'RationalQuadratic': ConstantKernel()*RationalQuadratic(), \
              'ExpSineSquared': ConstantKernel()*ExpSineSquared(), \
              'DotProduct': ConstantKernel()*DotProduct()}

# from .BRR_scikit import BRR_sk
from sklearn.linear_model import BayesianRidge

# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# # from tensorflow.keras.optimizers.legacy import Adam
# from tensorflow.keras.optimizers import Adam

import refann as rf

import corner

from numpy.random import multivariate_normal as MN

from .metrics import *

import time


# 0 IMPORT/SETUP REAL DATA

# 0.1 cosmic chronometers
# cc_loc = '../../datasets/Hdz_2020_CConly.txt'
cc_loc = 'https://raw.githubusercontent.com/reggiebernardo/datasets/main/Hdz_2020_CConly.txt'
cc_data = np.loadtxt(cc_loc)

z_cc = cc_data[:, 0]; Hz_cc = cc_data[:, 1]; sigHz_cc = cc_data[:, 2]


# 0.2 supernovae pantheon+
# pantheon plus systematics
# loc_lcparam = '../../datasets/pantheon2/Pantheon+SH0ES.dat'
loc_lcparam = 'https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/\
Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat'
lcparam = np.loadtxt(loc_lcparam, skiprows = 1, usecols = (2, 10, 11))

z_pp = lcparam[:, 0][111:]; muz_pp = lcparam[:, 1][111:]; errmuz_pp = lcparam[:, 2][111:]

# # load the pantheon+ covariance matrix
# loc_lcparam_sys = 'https://raw.githubusercontent.com/PantheonPlusSH0ES/\
# DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov'
# lcparam_sys = np.loadtxt(loc_lcparam_sys, skiprows = 1)

# # pantheon plus systematics
# cov = lcparam_sys.reshape(1701, 1701)
# cov_zred = cov[111:, 111:]
# cov_inv = np.linalg.inv(cov_zred)


# 0.3 desi year 1 bao data
# desi data: table 1 of 2404.03002
desi1_bgs_DVrd = [0.295, 7.93, 0.15]
desi1_lrg1_DMrdDHrd=[0.510, 13.62, 0.25, 20.98, 0.61, -0.445]
desi1_lrg2_DMrdDHrd=[0.706, 16.85, 0.32, 20.08, 0.60, -0.420]
desi1_lrg3_DMrdDHrd=[0.930, 21.71, 0.28, 17.88, 0.35, -0.389]
desi1_elg2_DMrdDHrd=[1.317, 27.79, 0.69, 13.82, 0.42, -0.444]
desi1_qso_DVrd = [1.491, 26.07, 0.67]
desi1_lya_DMrdDHrd = [2.330, 39.71, 0.94, 8.52, 0.17, -0.477]

def get_DVrd_fromDMDH(x):
    z, A, dA, B, dB, r = x
    f=(z*(A**2)*B)**(1/3)
    dfdA = (2/3)*A*B*z/(f**2)
    dfdB = f/(3*B)
    cov_AB = r * dA * dB
    delta_f = np.sqrt((dfdA * dA)**2 + (dfdB * dB)**2 + 2 * dfdA * dfdB * cov_AB)
    return [z, f, delta_f]

desi1_lrg1_DVrd = get_DVrd_fromDMDH(desi1_lrg1_DMrdDHrd)
desi1_lrg2_DVrd = get_DVrd_fromDMDH(desi1_lrg2_DMrdDHrd)
desi1_lrg3_DVrd = get_DVrd_fromDMDH(desi1_lrg3_DMrdDHrd)
desi1_elg2_DVrd = get_DVrd_fromDMDH(desi1_elg2_DMrdDHrd)
desi1_lya_DVrd = get_DVrd_fromDMDH(desi1_lya_DMrdDHrd)

# combining desi data into one array DV/rd
combined_array = []
for a, b, c, d, e, f, g in zip(desi1_bgs_DVrd, desi1_lrg1_DVrd, \
                               desi1_lrg2_DVrd, desi1_lrg3_DVrd, \
                               desi1_elg2_DVrd, desi1_qso_DVrd, desi1_lya_DVrd):
    combined_array.append([a, b, c, d, e, f, g])
combined_array_transposed = list(map(list, zip(*combined_array)))
desi1_DVrd=np.array(combined_array_transposed)

z_desi1=desi1_DVrd[:, 0]; DVrdz_desi1=desi1_DVrd[:, 1]; DVrdzERR_desi1=desi1_DVrd[:, 2]


# 0.4 RSD growth date in 1803.01337
# load growth RSD data
# loc_rsd = '../datasets/growth_1803.01337/Growth_tableII.txt'
loc_rsd='https://raw.githubusercontent.com/reggiebernardo/datasets/refs/heads/main/Growth_tableII.txt'
rsd = np.loadtxt(loc_rsd, usecols = (0, 1, 2))

z_rsd = rsd[:, 0]; fs8_rsd = rsd[:, 1]; sigfs8_rsd = rsd[:, 2]

# # the WiggleZ points (3x3) covariance
# # wgz_loc = '../datasets/growth_1803.01337/Cij_WiggleZ.txt'
# wgz_loc='https://raw.githubusercontent.com/reggiebernardo/datasets/refs/heads/main/Cij_WiggleZ.txt'
# C_wgz = np.loadtxt(wgz_loc)

# # f*sigma_8 data covariance with WiggleZ
# C_rsd_1 = np.diag(sigfs8_rsd[: 9]**2)
# C_rsd_w = C_wgz
# C_rsd_N = np.diag(sigfs8_rsd[12:]**2)

# C_rsd = block_diag(C_rsd_1, C_rsd_w, C_rsd_N)
# C_rsd_inv = np.linalg.inv(C_rsd)


# 0.xxx some helpful functions
def dVrdz_apy(z, cosmo, rd_fid=147.46):
    # cosmo=astropy cosmology object
    # example: cosmo = w0waCDM(H0=H0, Om0=Om0, Ode0=1-Om0-Ok0, w0=w0, wa=wa, Tcmb0=2.725)
    # rd_fid=147.46 # see 2406.05049
    dHz=c_kms/cosmo.H(z).value # parallel to line-of-sight
    dMz=cosmo.luminosity_distance(z).value/(1+z) # transverse to line-of-sight
    return ((z*(dMz**2)*dHz)**(1/3))/(rd_fid)


# 0.1.1 cosmological perturbations---analytical
# see arxiv:1505.06601 Nesseris & Sapone

def dltz_apy(z, cosmo, de_model='no pert', k2cs2=1e-10):
    # cosmo=Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa, Tcmb0=Tcmb0)
    # cosmo is an astropy cosmology object
    w0=cosmo.w0
    if de_model=='no pert':
        dB=0
    if de_model=='static':
        dB=(1 + w0)/(1 - 3*w0)
    if de_model=='dynamic':
        Om0=cosmo.Om0
        H0=cosmo.H0.value
        dB=(36/24)*Om0*(1+w0)*(H0**2)/k2cs2
    B=np.sqrt( ((1-3*w0)**2) + 24*dB )/(12*w0)
    alpha=(1/4) - (5/(12*w0)) + B
    beta=(1/4) - (5/(12*w0)) - B
    gamma=(1/2) + alpha + beta

    a = 1/(1 + z)
    Omz=cosmo.Om(z)
    return a*hyp2f1(alpha, beta, gamma, 1 - (1/Omz))

def fz_apy(z, cosmo, de_model='no pert', k2cs2=1e-10, hh=1e-10):
    # cosmo=Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa, Tcmb0=Tcmb0)
    # cosmo is an astropy cosmology object
    w0=cosmo.w0
    if de_model=='no pert':
        dB=0
    if de_model=='static':
        dB=(1 + w0)/(1 - 3*w0)
    if de_model=='dynamic':
        Om0=cosmo.Om0
        H0=cosmo.H0.value
        dB=(36/24)*Om0*(1+w0)*(H0**2)/k2cs2
    B=np.sqrt( ((1-3*w0)**2) + 24*dB )/(12*w0)
    alpha=(1/4) - (5/(12*w0)) + B
    beta=(1/4) - (5/(12*w0)) - B
    gamma=(1/2) + alpha + beta

    Omz=cosmo.Om(z)

    # approximate Om derivative by central difference
    # add factor to convert to derivative wrt to a
    dOmda=-((1+z)**2)*(cosmo.Om(z + hh) - cosmo.Om(z - hh))/(2*hh)

    a = 1/(1 + z)
    prefactor=a*alpha*beta/gamma
    hypergeom_factor=hyp2f1(alpha+1, beta+1, gamma+1, 1 - (1/Omz))/hyp2f1(alpha, beta, gamma, 1 - (1/Omz))
    Om_factor=dOmda/(Omz**2)

    return 1 + prefactor*hypergeom_factor*Om_factor

def fs8z_apy(z, cosmo, s8, de_model='no pert', k2cs2=1e-10, hh=1e-10):
    Om0=cosmo.Om0
    sigma8_0=s8*np.sqrt( 0.3/Om0 )
    dlt_0=dltz_apy(0, cosmo, de_model=de_model, k2cs2=k2cs2)
    dltz=dltz_apy(z, cosmo, de_model=de_model, k2cs2=k2cs2)
    
    s8z=sigma8_0*dltz/dlt_0
    fz=fz_apy(z, cosmo, de_model=de_model, k2cs2=k2cs2, hh=hh)
    return fz*s8z


# 0.yyy utils

def split_data(x, y, yerr, test_frac=0.1, random_state=14000605):
    '''break data into training and test sets'''
    X = np.column_stack((x, y, yerr))  # Stack x, y, and yerr into a single array
    X_train, X_test = train_test_split(X, test_size=test_frac, random_state=random_state)
    
    # Extract train data
    x_train = X_train[:, 0]; y_train = X_train[:, 1]; yerr_train = X_train[:, 2]
    train_dict = {'x': x_train, 'y': y_train, 'yerr': yerr_train}
    
    # Extract test data
    x_test = X_test[:, 0]; y_test = X_test[:, 1]; yerr_test = X_test[:, 2]
    test_dict = {'x': x_test, 'y': y_test, 'yerr': yerr_test}
    
    return {'train': train_dict, 'test': test_dict}

def how_to_mock(z_real, yerr_real, cosmo_func):
    # generate mock data
    yz_input = cosmo_func(z_real) # input cosmology/true curve; points centered around this line
    # yzs = []
    # for i in range(ndraws):
    #     yz_i = np.random.normal(loc=yz_input, scale=yerr_real, size=len(z_real))
    #     yzs.append(yz_i)
    # y_mock_0 = np.array(yzs)

    # # second line after np.mean(...) is an offset; otherwise, data will be on the true cruve
    # y_mock = np.array([np.mean(y_mock_0[:, i]) \
    #                    + np.random.normal(loc=0, scale=np.abs(np.mean(y_mock_0[:, i]) - y_real[i])) \
    #                    for i in np.arange(len(z_real))])
    # yerr_mock = np.array([np.std(y_mock_0[:, i]) for i in np.arange(len(z_real))])

    # mean
    y_mock=np.random.normal(loc=yz_input, scale=yerr_real, size=len(z_real))
    yerr_mock=yerr_real
    return z_real, y_mock, yerr_mock

def create_whisker_plot(ax, means, uncertainties, colors, linestyles, markers, labels, \
                        xlabel = r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', reference_value=None, \
                        linewidth=2, markersize=40, alpha=0.7):
    # fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    for mean, uncertainty in zip(means, uncertainties):
        method_data = np.random.normal(mean, uncertainty, 10000)
        data.append(method_data)

    for i, (method_data, linestyle, color, marker, label) in enumerate(zip(data, linestyles, colors, markers, labels)):
        mean_val = np.mean(method_data)
        conf_int = np.percentile(method_data, [16, 84])
        ax.hlines(y=i + 1, xmin=conf_int[0], xmax=conf_int[1], colors=color, \
                  linewidth=linestyles, linestyle=linestyle, alpha=alpha)
        ax.scatter(mean_val, i + 1, color=color, marker=marker, s=markersize, \
                   label=label, alpha=alpha)

    if reference_value is not None:
        ax.axvline(x=reference_value, color='gray', linestyle=':')

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_yticks(np.arange(1, len(data) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.legend()

def add_corner(chains, labels, fig=None, color='red', ls='-', lw=2, alpha=0.75, \
               add_truth=None, truth_color='gray', range=None, quantiles=None, plot_density=True):
    return corner.corner(chains, labels=labels, color=color, \
                         fig=fig, hist_kwargs={'linestyle': ls, 'linewidth': lw, 'alpha': alpha, 'density': True}, \
                         plot_datapoints=False, fill_contours=False, \
                         smooth=True, plot_density=plot_density, quantiles=quantiles, \
                         truths=add_truth, truth_color=truth_color, range=range, levels=(0.68,0.95,))


# 0.zzz methods: mcmc, ga, pyabc, etc.

def chi2(x, y, yerr, model_func):
    y_model=model_func(x)
    dev=(y-y_model)/yerr
    return np.sum( dev**2 )

def run_mcmc(ndim, nwalkers, nburn, nmcmc, dres, llprob, p0):
    with Pool() as pool:
        start = time.time()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, llprob, pool=pool)

        pos0 = [p0 + dres * np.random.randn(ndim) for i in range(nwalkers)]

        print("Running MCMC...")
        pos1 = sampler.run_mcmc(pos0, nburn, rstate0=np.random.get_state())
        sampler.reset()
        pos2 = pos1
        sampler.run_mcmc(pos2, nmcmc, rstate0=np.random.get_state(), progress=True)
        print("Done.")

        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("Total time:", time.time() - start)

        samps = sampler.chain[:, nburn:, :].reshape((-1, ndim))

        # Compute percentiles for each parameter
        param_percentiles = np.percentile(samps, [16, 50, 84], axis=0)

        # Print MCMC results
        print("MCMC result:")
        for i, (lower, median, upper) in enumerate(param_percentiles.T):
            lower_err = median - lower
            upper_err = upper - median
            print(f"    x[{i}] = {median} + {upper_err} - {lower_err}")

        return samps

def mcreconstruct_function(func, samples, x_rec, nmc=1000):
    # monte carlo reconstruction
    # setup mean and cov of samples
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)

    func_samples=[]
    for j in range(nmc):
        while True:
            params_j = MN(mean, cov)
            func_j = func(x_rec, params_j)
            if not np.any(np.isnan(func_j)):
                break
        func_samples.append( func_j )
        
    func_samples = np.array(func_samples)
    func_mean = np.mean(func_samples, axis=0)
    func_err = np.std(func_samples, axis=0)
    return x_rec, func_mean, func_err


# # BRR class

class BRR_sk:
    def __init__(self, n_order, tol=1e-6, init=[1., 1e-3]):
        """
        Initialize Bayesian Ridge Regression model.

        Parameters:
        - n_order (int): The order of the polynomial features.
        - tol (float): Tolerance for stopping criteria.
        - init (list): Initial values for alpha and lambda.
        """
        self.n_order = n_order
        self.tol = tol
        self.init = init

    def train(self, x_train, y_train, yerr_train):
        """
        Train the Bayesian Ridge Regression model.
    
        Parameters:
        - x_train (array-like): Training input data.
        - y_train (array-like): Target values.
        - yerr_train (array-like): Uncertainty of the target values.
    
        Returns:
        - reg0 (BayesianRidge): Trained Bayesian Ridge Regression model.
        """
        X_train = np.vander(x_train, N=self.n_order + 1, increasing=True)
        sample_weights = 1 / yerr_train**2
        sample_weights /= np.sum(sample_weights)
        
        reg0 = BayesianRidge(tol=self.tol, fit_intercept=False, compute_score=False)
        reg0.set_params(alpha_init=self.init[0], lambda_init=self.init[1])
        reg0.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Store the trained model in the brr attribute
        self.brr = reg0
        
        return reg0

    def predict(self, x_test):
        """
        Predict target values and uncertainties for test data.

        Parameters:
        - x_test (array-like): Test input data.

        Returns:
        - predictions (dict): Dictionary containing predicted values and uncertainties.
        """
        # Assuming x_train, y_train, yerr_train are attributes of the class
        X_test = np.vander(x_test, self.n_order + 1, increasing=True)
        ymean, ystd = self.brr.predict(X_test, return_std=True)
        return {'z': x_test, 'Y': ymean, 'varY': ystd**2}


# # 1 GW COSMO CLASS

class CosmoLearn:
    # def __init__(self, params, de_model='no pert', k2cs2=1e-10, hh=1e-10, \
    #              pop_model='Pop III', rd_fid=147.46, Tcmb0=2.725, seed=None):
    def __init__(self, params, de_model='no pert', k2cs2=1e-10, hh=1e-10, \
                 rd_fid=147.46, Tcmb0=2.725, seed=None):
        self.params = params
        self.de_model=de_model
        self.k2cs2=k2cs2
        self.hh=hh
        # self.pop_model=pop_model
        self.rd_fid=rd_fid
        self.Tcmb0=Tcmb0
        self.seed = seed
        if self.seed is not None:
            # np.random.default_rng(self.seed)
            np.random.seed(self.seed)
        self.mock_data={}

        # ml defaults
        self.ga_params={'max_num_iteration': 1000, 'population_size':100, \
                        'mutation_probability': 0.25, 'elit_ratio': 0.01, \
                        'crossover_probability': 0.5, 'parents_portion': 0.3, \
                        'crossover_type':'uniform', 'max_iteration_without_improv': None}

    #### 1.1 Mock Data Generation

    def set_cosmo(self):
        # H0, Om0, _, w0, _, _=self.params
        H0, Om0, w0, _=self.params
        Tcmb0=self.Tcmb0
        Ok0=0; wa=0
        self.cosmo=w0waCDM(H0=H0, Om0=Om0, Ode0=1-Om0-Ok0, w0=w0, wa=wa, Tcmb0=Tcmb0)

    def cosmo_input(self, x, key='CosmicChronometers'):
        H0, Om0, w0, s8=self.params
        Ok0=0; wa=0
        if key=='CosmicChronometers':
            return self.cosmo.H(x).value
        if key=='RedshiftSpaceDistorsions':
            return fs8z_apy(x, self.cosmo, s8, de_model=self.de_model, k2cs2=self.k2cs2, hh=self.hh)
        if key=='SuperNovae':
            return self.cosmo.distmod(x).value
        if key=='BaryonAcousticOscillations':
            return dVrdz_apy(x, self.cosmo, rd_fid=self.rd_fid)
        if key=='BrightSirens':
            return self.cosmo.luminosity_distance(x).value/1000

    def make_cosmic_chronometers_like(self, how_to_mock=how_to_mock):
        # generate mock data
        # Hz_samples = self.cosmo.H(z_cc).value
        # z_mock, y_mock, yerr_mock=\
        # how_to_mock(z_cc, sigHz_cc, cosmo_func=lambda x: self.cosmo.H(x).value)
        key='CosmicChronometers'
        z_mock, y_mock, yerr_mock=\
        how_to_mock(z_cc, sigHz_cc, cosmo_func=lambda x: self.cosmo_input(x, key=key))
        self.mock_data[key]=split_data(z_mock, y_mock, yerr_mock, random_state=self.seed)
        return z_mock, y_mock, yerr_mock

    def fs8z(self, z):
        # cosmo=astropy cosmology object
        # example: cosmo = w0waCDM(H0=H0, Om0=Om0, Ode0=1-Om0-Ok0, w0=w0, wa=wa, Tcmb0=2.725)
        _, _, _, s8=self.params
        return fs8z_apy(z, self.cosmo, s8, de_model=self.de_model, k2cs2=self.k2cs2, hh=self.hh)

    def make_rsd_like(self, how_to_mock=how_to_mock):
        # generate mock data: growth rate rsd
        # fs8z_samples = self.fs8z(z_rsd, de_model=de_model, k2cs2=k2cs2, hh=hh)
        # z_mock, y_mock, yerr_mock=how_to_mock(z_rsd, sigfs8_rsd, cosmo_func=lambda x: self.fs8z(x))
        key='RedshiftSpaceDistorsions'
        z_mock, y_mock, yerr_mock=how_to_mock(z_rsd, sigfs8_rsd, \
                                              cosmo_func=lambda x: self.cosmo_input(x, key=key))
        self.mock_data[key]=split_data(z_mock, y_mock, yerr_mock, random_state=self.seed)
        return z_mock, y_mock, yerr_mock

    def make_pantheon_plus_like(self, how_to_mock=how_to_mock):
        # generate mock data
        # muz_samples = self.cosmo.distmod(z_pp).value
        # z_mock, y_mock, yerr_mock=\
        # how_to_mock(z_pp, errmuz_pp, cosmo_func=lambda x: self.cosmo.distmod(x).value)
        key='SuperNovae'
        z_mock, y_mock, yerr_mock=\
        how_to_mock(z_pp, errmuz_pp, cosmo_func=lambda x: self.cosmo_input(x, key=key))
        self.mock_data[key]=split_data(z_mock, y_mock, yerr_mock, random_state=self.seed)
        return z_mock, y_mock, yerr_mock
    
    def dVrdz(self, z):
        # cosmo=astropy cosmology object
        # example: cosmo = w0waCDM(H0=H0, Om0=Om0, Ode0=1-Om0-Ok0, w0=w0, wa=wa, Tcmb0=2.725)
        # rd_fid=147.46 # see 2406.05049 
        dHz=c_kms/self.cosmo.H(z).value # parallel to line-of-sight
        dMz=self.cosmo.luminosity_distance(z).value/(1+z) # transverse to line-of-sight
        return ((z*(dMz**2)*dHz)**(1/3))/(self.rd_fid)

    def make_desi1_like(self, how_to_mock=how_to_mock):
        # generate mock data
        # dVrdz_samples = self.dVrdz(z_desi1)
        # z_mock, y_mock, yerr_mock=\
        # how_to_mock(z_desi1, DVrdzERR_desi1, cosmo_func=lambda x: self.dVrdz(x))
        key='BaryonAcousticOscillations'
        z_mock, y_mock, yerr_mock=\
        how_to_mock(z_desi1, DVrdzERR_desi1, cosmo_func=lambda x: self.cosmo_input(x, key=key))
        self.mock_data[key]=split_data(z_mock, y_mock, yerr_mock, random_state=self.seed)
        return z_mock, y_mock, yerr_mock
    
    def make_bright_sirens_mock(self, years=3, pop_model='Pop III'):
        # pop_models=['Pop III', 'Delay', 'No Delay']
        H0, Om0, w0, _=self.params
        Ok0=0; wa=0
        params_w0wa=[H0, Om0, Ok0, w0, wa] # input to LISA code is w0wa params: H0, Om0, Ok0, w0, wa
        # z_mock, y_mock, yerr_mock = generate(population=self.pop_model, years=years, params=params_w0wa)
        z_mock, y_mock, yerr_mock = generate(population=pop_model, years=years, params=params_w0wa)
        self.mock_data['BrightSirens']=split_data(z_mock, y_mock, yerr_mock, random_state=self.seed)
        return z_mock, y_mock, yerr_mock

    def make_mock(self, mock_keys, pop_model='Pop III', years=5):
        self.set_cosmo()
        for key in mock_keys:
            if key=='CosmicChronometers':
                self.make_cosmic_chronometers_like()
            if key=='SuperNovae':
                self.make_pantheon_plus_like()
            if key=='BaryonAcousticOscillations':
                self.make_desi1_like()
            if key=='BrightSirens':
                self.make_bright_sirens_mock(pop_model=pop_model, years=years)
            if key=='RedshiftSpaceDistorsions':
                self.make_rsd_like()

    def plot_train_test_data(self, ax, Data_Key, alpha=0.7, markersize=3, \
                             fmt_train='go', label_train='Training Set', fmt_test='ms', label_test='Test Set'):
        if Data_Key != 'BaryonAcousticOscillations':
            Cosmo_Data=self.mock_data[Data_Key]
            ax.errorbar(Cosmo_Data['train']['x'], Cosmo_Data['train']['y'], yerr=Cosmo_Data['train']['yerr'], \
                        markersize=markersize, fmt=fmt_train, alpha=alpha, label=label_train)
            ax.errorbar(Cosmo_Data['test']['x'], Cosmo_Data['test']['y'], yerr=Cosmo_Data['test']['yerr'], \
                        markersize=markersize, fmt=fmt_test, alpha=alpha, label=label_test)
        if Data_Key == 'BaryonAcousticOscillations':
            Cosmo_Data=self.mock_data[Data_Key]
            ax.errorbar(Cosmo_Data['train']['x'], Cosmo_Data['train']['y']/(Cosmo_Data['train']['x']**(2/3)), \
                        yerr=Cosmo_Data['train']['yerr']/(Cosmo_Data['train']['x']**(2/3)), \
                        markersize=markersize, fmt=fmt_train, alpha=alpha, label=label_train)
            ax.errorbar(Cosmo_Data['test']['x'], Cosmo_Data['test']['y']/(Cosmo_Data['test']['x']**(2/3)), \
                        yerr=Cosmo_Data['test']['yerr']/(Cosmo_Data['test']['x']**(2/3)), \
                        markersize=markersize, fmt=fmt_test, alpha=alpha, label=label_test)

    def plot_residuals(self, ax, Data_Key, markersize=3, fmt_train='go', fmt_test='ms', alpha=0.7):
        Cosmo_Data = self.mock_data[Data_Key]
        ax.errorbar(Cosmo_Data['train']['x'], \
                    Cosmo_Data['train']['y'] - self.cosmo_input(Cosmo_Data['train']['x'], key=Data_Key), 
                    yerr=Cosmo_Data['train']['yerr'], markersize=markersize, fmt=fmt_train, alpha=alpha)
        ax.errorbar(Cosmo_Data['test']['x'], \
                    Cosmo_Data['test']['y'] - self.cosmo_input(Cosmo_Data['test']['x'], key=Data_Key), 
                    yerr=Cosmo_Data['test']['yerr'], markersize=markersize, fmt=fmt_test, alpha=alpha)

    def show_mocks(self, ax=None, show_input=False, figsize=(10, 10), markersize=3, \
                   fmt_train='go', label_train='Training Set', fmt_test='ms', label_test='Test Set', \
                   fmt_input='k-', label_input='Input', alpha_all=0.7, alpha_sne=0.1):
        fig = None  # initialize fig to None
        if ax is None:
            fig, ax = plt.subplots(nrows=len(self.mock_data.keys()), figsize=figsize)
        
        for i, key in enumerate(self.mock_data.keys()):
            if key == 'CosmicChronometers':
                self.plot_train_test_data(ax[i], key, alpha=alpha_all, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i].set_ylabel(r'$H(z)$')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i].plot(x_space, self.cosmo_input(x_space, key=key), fmt_input, alpha=alpha_all, label=label_input)
            elif key == 'SuperNovae':
                self.plot_train_test_data(ax[i], key, alpha=alpha_sne, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i].set_ylabel(r'$\mu(z)$')
                ax[i].set_xscale('log')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.logspace(np.log10(min(x_train)), np.log10(max(x_train)))
                    ax[i].plot(x_space, self.cosmo_input(x_space, key=key), fmt_input, alpha=alpha_all, label=label_input)
            elif key == 'BaryonAcousticOscillations':
                self.plot_train_test_data(ax[i], key, alpha=alpha_all, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i].set_ylabel(r'$D_{\rm V}/\left( r_{\rm D} z^{2/3} \right)$')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i].plot(x_space, self.cosmo_input(x_space, key=key)/(x_space**(2/3)), \
                               fmt_input, alpha=alpha_all, label=label_input)
            elif key == 'BrightSirens':
                self.plot_train_test_data(ax[i], key, alpha=alpha_all, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i].set_ylabel(r'$d_L(z)$')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i].plot(x_space, self.cosmo_input(x_space, key=key), fmt_input, alpha=alpha_all, label=label_input)
            elif key == 'RedshiftSpaceDistorsions':
                self.plot_train_test_data(ax[i], key, alpha=alpha_all, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i].set_ylabel(r'$f \sigma_8(z)$')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i].plot(x_space, self.cosmo_input(x_space, key=key), fmt_input, alpha=alpha_all, label=label_input)

        ax[-1].set_xlabel(r'Redshift $z$')

        # return fig and ax only if a new figure was created (i.e., if ax was None)
        if fig is not None:
            return fig, ax
        else:
            return None  # no return when ax is passed

    def show_mocks_and_residuals(self, ax=None, show_input=False, figsize=(10, 10), markersize=3, \
                                 fmt_train='go', label_train='Training Set', fmt_test='ms', label_test='Test Set', \
                                 ls_input='-', color_input='k', label_input='Input', alpha_all=0.7, alpha_sne=0.1, \
                                 input_zorder=1000000):
        fig = None  # initialize fig to None
        if ax is None:
            fig, ax = plt.subplots(nrows=len(self.mock_data.keys()), ncols=2, figsize=figsize)
        
        for i, key in enumerate(self.mock_data.keys()):
            if key == 'CosmicChronometers':
                self.plot_train_test_data(ax[i,0], key, alpha=alpha_all, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i,0].set_ylabel(r'$H(z)$')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,0].plot(x_space, self.cosmo_input(x_space, key=key), \
                                 ls=ls_input, color=color_input, alpha=alpha_all, label=label_input)

                self.plot_residuals(ax[i,1], key, markersize=markersize, \
                                    fmt_train=fmt_train, fmt_test=fmt_test, alpha=alpha_all)

                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,1].axhline(0, ls=ls_input, color=color_input, alpha=alpha_all)
                ax[i,1].yaxis.tick_right()

            elif key == 'SuperNovae':
                self.plot_train_test_data(ax[i,0], key, alpha=alpha_sne, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i,0].set_ylabel(r'$\mu(z)$')
                ax[i,0].set_xscale('log'); ax[i,1].set_xscale('log')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.logspace(np.log10(min(x_train)), np.log10(max(x_train)))
                    ax[i,0].plot(x_space, self.cosmo_input(x_space, key=key), \
                                 ls=ls_input, color=color_input, alpha=alpha_all, label=label_input)

                self.plot_residuals(ax[i,1], key, markersize=markersize, \
                                    fmt_train=fmt_train, fmt_test=fmt_test, alpha=alpha_sne)
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,1].axhline(0, ls=ls_input, color=color_input, alpha=alpha_all)
                ax[i,1].yaxis.tick_right()
                        
            elif key == 'BaryonAcousticOscillations':
                self.plot_train_test_data(ax[i,0], key, alpha=alpha_all, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i,0].set_ylabel(r'$D_{\rm V}/\left( r_{\rm D} z^{2/3} \right)$')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,0].plot(x_space, self.cosmo_input(x_space, key=key)/(x_space**(2/3)), \
                                 ls=ls_input, color=color_input, alpha=alpha_all, label=label_input)

                self.plot_residuals(ax[i,1], key, markersize=markersize, \
                                    fmt_train=fmt_train, fmt_test=fmt_test, alpha=alpha_all)

                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,1].axhline(0, ls=ls_input, color=color_input, alpha=alpha_all)
                ax[i,1].yaxis.tick_right()

            elif key == 'BrightSirens':
                self.plot_train_test_data(ax[i,0], key, alpha=alpha_all, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i,0].set_ylabel(r'$d_L(z)$')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,0].plot(x_space, self.cosmo_input(x_space, key=key), \
                                 ls=ls_input, color=color_input, alpha=alpha_all, label=label_input)

                self.plot_residuals(ax[i,1], key, markersize=markersize, \
                                    fmt_train=fmt_train, fmt_test=fmt_test, alpha=alpha_all)

                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,1].axhline(0, ls=ls_input, color=color_input, alpha=alpha_all)
                ax[i,1].yaxis.tick_right()

            elif key == 'RedshiftSpaceDistorsions':
                self.plot_train_test_data(ax[i,0], key, alpha=alpha_all, markersize=markersize, \
                                          fmt_train=fmt_train, label_train=label_train, \
                                          fmt_test=fmt_test, label_test=label_test)
                ax[i,0].set_ylabel(r'$f \sigma_8(z)$')
                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,0].plot(x_space, self.cosmo_input(x_space, key=key), \
                                 ls=ls_input, color=color_input, alpha=alpha_all, label=label_input)

                self.plot_residuals(ax[i,1], key, markersize=markersize, \
                                    fmt_train=fmt_train, fmt_test=fmt_test, alpha=alpha_all)

                if show_input:
                    x_train=self.mock_data[key]['train']['x']
                    x_space=np.linspace(min(x_train), max(x_train))
                    ax[i,1].axhline(0, ls=ls_input, color=color_input, \
                                    alpha=alpha_all, zorder=input_zorder)
                ax[i,1].yaxis.tick_right()

        ax[-1,0].set_xlabel(r'Redshift $z$'); ax[-1,1].set_xlabel(r'Redshift $z$')
        ax[0,0].set_title(r'Mock Observable'); ax[0,1].set_title(r'Residuals')

        # return fig and ax only if a new figure was created (i.e., if ax was None)
        if fig is not None:
            return fig, ax
        else:
            return None  # no return when ax is passed


    #### 1.2 Likelihoods

    def cosmo_func_wcdm(self, x, model_params, key='CosmicChronometers'):
        H0, Om0, w0, s8, rd_fid=model_params
        Ok0=0; wa=0
        cosmo_model=w0waCDM(H0=H0, Om0=Om0, Ode0=1-Om0-Ok0, w0=w0, wa=wa, Tcmb0=self.Tcmb0)
        if key=='CosmicChronometers':
            return cosmo_model.H(x).value
        if key=='RedshiftSpaceDistorsions':
            return fs8z_apy(x, cosmo_model, s8, de_model=self.de_model, k2cs2=self.k2cs2, hh=self.hh)
        if key=='SuperNovae':
            return cosmo_model.distmod(x).value
        if key=='BaryonAcousticOscillations':
            return dVrdz_apy(x, cosmo_model, rd_fid=rd_fid)
        if key=='BrightSirens':
            return cosmo_model.luminosity_distance(x).value/1000

    def loglike_wcdm(self, model_params, Tcmb0=None, de_model='no pert', k2cs2=1e-10, hh=1e-10):
        ll=0
        for key, val in self.mock_data.items():
            train_data=val['train']
            x, y, yerr=train_data['x'], train_data['y'], train_data['yerr']
            if key=='CosmicChronometers':
                ll += -0.5*chi2(x, y, yerr, model_func=lambda x: self.cosmo_func_wcdm(x, model_params, key=key))
            if key=='RedshiftSpaceDistorsions':
                ll += -0.5*chi2(x, y, yerr, model_func=lambda x: self.cosmo_func_wcdm(x, model_params, key=key))
            if key=='SuperNovae':
                ll += -0.5*chi2(x, y, yerr, model_func=lambda x: self.cosmo_func_wcdm(x, model_params, key=key))
            if key=='BaryonAcousticOscillations':
                ll += -0.5*chi2(x, y, yerr, model_func=lambda x: self.cosmo_func_wcdm(x, model_params, key=key))
            if key=='BrightSirens':
                ll += -0.5*chi2(x, y, yerr, model_func=lambda x: self.cosmo_func_wcdm(x, model_params, key=key))

        return ll

    def lnprior_wcdm(self, x, \
                     prior_dict={'H0_min': 0, 'H0_max': 100, 'Om0_min': 0, 'Om0_max': 1, \
                                 'w0_min': -10, 'w0_max': 10, 's8_min': 0.2, 's8_max': 1.5}, \
                     rd_fid_prior={'mu': 147.46, 'sigma': 0.28}):
        H0, Om0, w, s8, rd_fid=x

        # flat priors
        H0_min=prior_dict['H0_min']; H0_max=prior_dict['H0_max']
        Om0_min=prior_dict['Om0_min']; Om0_max=prior_dict['Om0_max']
        w0_min=prior_dict['w0_min']; w0_max=prior_dict['w0_max']
        s8_min=prior_dict['s8_min']; s8_max=prior_dict['s8_max']
        if not (H0_min < H0 < H0_max and Om0_min < Om0 < Om0_max \
                and w0_min < w < w0_max and s8_min < s8 < s8_max):
            return -np.inf

        # gaussian prior on rd
        mu = rd_fid_prior['mu']; sigma = rd_fid_prior['sigma']
        prior_gauss=np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(rd_fid-mu)**2/sigma**2

        return prior_gauss

    def llprob_wcdm(self, x, \
                    prior_dict={'H0_min': 0, 'H0_max': 100, 'Om0_min': 0, 'Om0_max': 1, \
                                'w0_min': -10, 'w0_max': 10, 's8_min': 0.2, 's8_max': 1.5}, \
                    rd_fid_prior={'mu': 147.46, 'sigma': 0.28}, \
                    Tcmb0=None, de_model='no pert', k2cs2=1e-10, hh=1e-10):
        H0, Om0, w, s8, rd_fid=x
        if Tcmb0==None:
            Tcmb0=self.Tcmb0

        # check prior
        lp = self.lnprior_wcdm(x, prior_dict=prior_dict, rd_fid_prior=rd_fid_prior)
        if np.isfinite(lp)==False:
            return -np.inf
        
        lk = self.loglike_wcdm(x, Tcmb0=Tcmb0, de_model=de_model, k2cs2=k2cs2, hh=hh)
        if np.isnan(lk):
            return -np.inf
        return lp + lk


    #### 1.3 Methods/Training

    def get_mcmc_samples(self, nwalkers, dres, llprob, p0, nburn=100, nmcmc=500):
        print('Optimizing initial position...')
        p0_x=minimize(lambda x: -2*llprob(x), x0=p0, method='Nelder-Mead').x
        print('... At', p0_x)

        ndim=len(dres)
        mcmc_samples=run_mcmc(ndim, nwalkers, nburn, nmcmc, dres, llprob, p0_x)

        self.mcmc_samples=mcmc_samples
        return mcmc_samples

    def get_gaFisher_samples(self, fitness_func, prior, \
                             llprob=None, nsamples=10000, convergence_curve=False):

        ga_model=GA(function=fitness_func,dimension=len(prior),variable_type='real', \
                    variable_boundaries=np.array(prior), algorithm_parameters=self.ga_params, \
                    convergence_curve=convergence_curve)
        ga_model.run()
        self.ga_model=ga_model
        
        # surround GA best fit with covariance from Fisher matrix
        if llprob==None:
            llprob=lambda x: -0.5*fitness_func(x)

        ga_sol=self.ga_model.output_dict['variable']
        FisherM = -Hessian(llprob)(ga_sol)
        cov_FisherM = np.linalg.inv(FisherM)
        gaFisher_samples = np.random.multivariate_normal(ga_sol, cov_FisherM, size=nsamples)

        # compute percentiles for each parameter
        param_percentiles = np.percentile(gaFisher_samples, [16, 50, 84], axis=0)

        # print GA results
        print()
        print("GA-Fisher result:")
        for i, (lower, median, upper) in enumerate(param_percentiles.T):
            lower_err = median - lower
            upper_err = upper - median
            print(f"    x[{i}] = {median} + {upper_err} - {lower_err}")

        self.gaFisher_samples=gaFisher_samples
        return gaFisher_samples

    def train_gp(self, kernel_key='RBF', n_restarts_optimizer = 10):
        kernel=kernels_sk[kernel_key]
        GP_dict={}
        for key in self.mock_data.keys():
            train_data=self.mock_data[key]['train']
            x=train_data['x']; y=np.column_stack((train_data['y'], train_data['yerr']))
            gp_cosmo = GaussianProcessRegressor(kernel = kernel, alpha = y[:, 1]**2, \
                                                n_restarts_optimizer = n_restarts_optimizer)
            if key != 'SuperNovae':
                gp_cosmo.fit(x.reshape(-1, 1), y[:, 0]);
            if key == 'SuperNovae':
                gp_cosmo.fit(np.log10(x).reshape(-1, 1), y[:, 0]);
            GP_dict[key]=gp_cosmo
        self.GP_dict=GP_dict

    def train_brr(self, n_order = 3):
        BRR_dict={}
        for key in self.mock_data.keys():
            train_data=self.mock_data[key]['train']
            x=train_data['x']; y=np.column_stack((train_data['y'], train_data['yerr']))
            brr_cosmo = BRR_sk(n_order = n_order)
            if key != 'SuperNovae':
                brr_cosmo.train(x, y[:, 0], y[:, 1]);
            if key == 'SuperNovae':
                brr_cosmo.train(np.log10(x), y[:, 0], y[:, 1]);
            BRR_dict[key]=brr_cosmo
        self.BRR_dict=BRR_dict

    # def init_ann(self, show_summary=False):
    #     ann_arch={}
    #     for key in self.mock_data.keys():
    #         if key == 'CosmicChronometers' or key =='RedshiftSpaceDistorsions' \
    #             or key == 'BaryonAcousticOscillations':
    #             ann=tf.keras.Sequential([Dense(32, activation='relu', input_shape=[1], \
    #                                            kernel_regularizer=tf.keras.regularizers.l2()), \
    #                                      Dropout(0.1), \
    #                                      Dense(64, activation='relu', \
    #                                            kernel_regularizer=tf.keras.regularizers.l2()), \
    #                                      Dropout(0.1), Dense(2),])
    #             ann.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    #             ann_arch[key]=ann
    #         if key == 'BrightSirens':
    #             ann=tf.keras.Sequential([Dense(128, activation='relu', input_shape=[1], \
    #                                            kernel_regularizer=tf.keras.regularizers.l2()), \
    #                                      Dropout(0.1), \
    #                                      Dense(128, activation='relu', \
    #                                            kernel_regularizer=tf.keras.regularizers.l2()), \
    #                                      Dropout(0.1), Dense(2),])
    #             ann.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    #             ann_arch[key]=ann
    #         if key == 'SuperNovae':
    #             ann=tf.keras.Sequential([Dense(256, activation='relu', input_shape=[1], \
    #                                            kernel_regularizer=tf.keras.regularizers.l2()), \
    #                                      Dropout(0.1), \
    #                                      Dense(512, activation='relu', \
    #                                            kernel_regularizer=tf.keras.regularizers.l2()), \
    #                                      Dropout(0.1), Dense(2),])
    #             ann.compile(optimizer=Adam(learning_rate=0.00015), loss=tf.keras.losses.MeanSquaredError())
    #             ann_arch[key]=ann

    #     self.ANN_arch=ann_arch
    #     if show_summary:
    #         for key in self.mock_data.keys():
    #             print(f'ANN-design for {key}')
    #             self.ANN_arch[key].summary()
    #             print()

    def init_ann(self, mid_node = 4096, hidden_layer = 1, hp_model = 'rec_1', \
                 loss_func='L1', iteration=30000, print_info = False, show_summary = False):
        ANN_dict={}
        for key in self.mock_data.keys():
            train_data=self.mock_data[key]['train']
            if key != 'SuperNovae': 
                data=np.column_stack((train_data['x'], train_data['y'], train_data['yerr']))
            if key == 'SuperNovae':
                data=np.column_stack((np.log10(train_data['x']), train_data['y'], train_data['yerr']))
                
            ann_cosmo = rf.ANN(data, mid_node=mid_node, hidden_layer=hidden_layer, \
                               hp_model=hp_model, loss_func=loss_func)
            ann_cosmo.print_info = print_info
            ann_cosmo.iteration = iteration
            ANN_dict[key]=ann_cosmo

        self.ANN_dict=ANN_dict

        if show_summary:
            for key in self.mock_data.keys():
                print(f'ANN-design for {key}')
                ann=self.ANN_dict[key]
                print('mid node:', ann.mid_node, 
                      'hidden layer:', ann.hidden_layer, 
                      'hp model:', ann.hp_model,
                      'loss function:', ann.loss_func,
                      'n_epochs:', ann.iteration,
                      'learning rate:', ann.lr,
                      'minimum learning rate:', ann.lr_min,
                      'max batch size:', ann.batch_size_max)
                print()

    def show_ann_summary(self, key=None):
        if key is None:
            for key in self.ANN_dict.keys():
                print(f'ANN-design for {key}')
                ann=self.ANN_dict[key]
                print('mid node:', ann.mid_node, 
                      'hidden layer:', ann.hidden_layer, 
                      'hp model:', ann.hp_model, 
                      'loss function:', ann.loss_func,
                      'n_epochs:', ann.iteration,
                      'learning rate:', ann.lr,
                      'minimum learning rate:', ann.lr_min,
                      'max batch size:', ann.batch_size_max)
                print()
        else:
            if key in self.ANN_dict.keys():
                ann=self.ANN_dict[key]
                print('mid node:', ann.mid_node, 
                      'hidden layer:', ann.hidden_layer, 
                      'hp model:', ann.hp_model, 
                      'loss function:', ann.loss_func,
                      'n_epochs:', ann.iteration,
                      'learning rate:', ann.lr,
                      'minimum learning rate:', ann.lr_min,
                      'max batch size:', ann.batch_size_max)
            else:
                print(f'ANN with key {key} not found.')

    # def train_ann(self, use_early_stop=True, epochs=10000, validation_split=0.1, verbose=0, patience=1000):
    #     ANN_dict={}
    #     early_stop=None
    #     if use_early_stop:
    #         early_stop=EarlyStopping(patience=patience, restore_best_weights=True)
    #     for key in self.mock_data.keys():
    #         print(f'ANN training w/ {key} data')
    #         train_data=self.mock_data[key]['train']
    #         x=train_data['x']; y=np.column_stack((train_data['y'], train_data['yerr']))
    #         ann_cosmo=self.ANN_arch[key]
    #         if key != 'SuperNovae':
    #             ann_cosmo.fit(x, y, epochs=epochs, validation_split=validation_split, \
    #                           callbacks=[early_stop], verbose=verbose)                
    #         if key == 'SuperNovae':
    #             ann_cosmo.fit(np.log10(x), y, epochs=epochs, validation_split=validation_split, \
    #                           callbacks=[early_stop], verbose=verbose)
    #         ann_hist=ann_cosmo.history.history  
    #         ANN_dict[key]={'ANN': ann_cosmo, 'loss': ann_hist['loss'], \
    #                        'val_loss': ann_hist['val_loss']}
    #     self.ANN_dict=ANN_dict

    def train_ann(self):
        ann=self.ANN_dict
        # ANN_dict={}
        for key in self.mock_data.keys():
            # print(f'ANN training w/ {key} data')
            # ann_cosmo=ann[key]
            # ann_cosmo.train()   

            start_time = time.time()
            ann_cosmo = ann[key]
            ann_cosmo.train()
            end_time = time.time()
            duration = end_time - start_time
            print(f'ANN training w/ {key} data completed in {duration:.2f} seconds')
        #     ANN_dict[key]={'ANN': ann_cosmo, 'loss': ann_cosmo.loss, 'n_epochs': ann_cosmo.iteration}
        # self.ANN_dict=ANN_dict

    # def show_ann_loss(self, ax=None, figsize=(10, 10)):
    #     fig = None  # initialize fig to None
    #     if ax is None:
    #         fig, ax = plt.subplots(nrows=len(self.mock_data.keys()), figsize=figsize)
        
    #     for i, key in enumerate(self.mock_data.keys()):
    #         n_epochs=len(self.ANN_dict[key]['loss'])
    #         ax[i].plot(self.ANN_dict[key]['loss'], 'g-', alpha=0.7, label=f'Training')
    #         ax[i].plot(self.ANN_dict[key]['val_loss'], '--', color='purple', alpha=0.7, label='Validation')
    #         ax[i].set_yscale('log'); ax[i].set_xscale('log'); ax[i].set_ylabel(f'Loss')
    #         ax[i].set_xlim(1, n_epochs); ax[i].legend(loc='lower left', prop={'size': 10})
    #         ax[i].get_legend().set_title(f'{key}')

    #     ax[-1].set_xlabel(r'Epoch')

    #     # return fig and ax only if a new figure was created (i.e., if ax was None)
    #     if fig is not None:
    #         return fig, ax
    #     else:
    #         return None  # no return when ax is passed  

    def show_ann_loss(self, ax=None, figsize=(10, 10)):
        fig = None  # initialize fig to None
        if ax is None:
            fig, ax = plt.subplots(nrows=len(self.mock_data.keys()), figsize=figsize)
        
        for i, key in enumerate(self.mock_data.keys()):
            ann_cosmo=self.ANN_dict[key]
            n_epochs=ann_cosmo.iteration
            ax[i].plot(range(n_epochs), ann_cosmo.loss, 'g-', alpha=0.7)
            ax[i].set_yscale('log'); ax[i].set_xscale('log'); ax[i].set_ylabel(f'Loss')
            ax[i].set_xlim(1, n_epochs); ax[i].legend(loc='lower left', prop={'size': 10})
            ax[i].get_legend().set_title(f'{key}')

        ax[-1].set_xlabel(r'Epoch')

        # return fig and ax only if a new figure was created (i.e., if ax was None)
        if fig is not None:
            return fig, ax
        else:
            return None  # no return when ax is passed  

    #### 1.4 Show Best Fits

    def show_param_posterior(self, fig=None, method='MCMC', show_rd=False, \
                             color='red', ls='-', lw=2, alpha=0.75, \
                             show_truth=False, truth_color='gray', range=None):
        # default labels
        labels = [r'$H_0$', r'$\Omega_{m0}$', r'$w$', r'$S_8=\sigma_8 \sqrt{\Omega_{m0}/0.3}$']
        if show_rd:
            labels.append(r'$r_{\rm D}$')  # add extra label if `show_rd` is True
        
        # select samples based on the method
        if method == 'MCMC':
            samples = self.mcmc_samples
        elif method == 'GAFisher':
            samples = self.gaFisher_samples
        else:
            raise ValueError("supported methods: 'MCMC' and 'GAFisher'")
        
        # if `show_rd` is False, exclude the last column (assumes `r_D` is the last parameter)
        if not show_rd:
            samples = samples[:, :4]  # Only use the first four columns

        # determine truths if requested
        add_truth = None
        if show_truth:
            add_truth = list(self.params)
            if show_rd:
                add_truth.append(self.rd_fid)

        # if `fig` is None, create a new corner plot
        if fig is None:
            corner_plot = add_corner(samples, labels, color=color, ls=ls, lw=lw, alpha=alpha, \
                                     add_truth=add_truth, truth_color=truth_color, range=range)
            return corner_plot
        else:
            # add to an existing figure without returning
            add_corner(samples, labels, fig=fig, color=color, ls=ls, lw=lw, alpha=alpha, \
                       add_truth=add_truth, truth_color=truth_color, range=range)

    def show_bestfit_curve(self, ax=None, figsize=(10, 10), method='MCMC', nmc=1000, \
                           n_sigma_rec=2, n_recz=100, color='red', alpha=0.25, hatch=None, label='None'):
        fig = None  # initialize fig to None
        if ax is None:
            fig, ax = plt.subplots(nrows=len(self.mock_data.keys()), figsize=figsize)
        
        for i, key in enumerate(self.mock_data.keys()):
            cosmo_func=lambda x, p: self.cosmo_func_wcdm(x, p, key=key)
            x_train=self.mock_data[key]['train']['x']
            if key=='SuperNovae':
                x_rec=np.logspace(np.log10(min(x_train)), np.log10(max(x_train)), n_recz)
            else:
                x_rec=np.linspace(min(x_train), max(x_train), n_recz)

            # parametric reconstruction---assuming Gaussian samples
            if method=='MCMC':
                samples=self.mcmc_samples
            if method=='GAFisher':
                samples=self.gaFisher_samples
            x_rec, func_mean, func_err=mcreconstruct_function(cosmo_func, samples, x_rec=x_rec, nmc=nmc)

            if key!='BaryonAcousticOscillations':
                ax[i].fill_between(x_rec, func_mean-n_sigma_rec*func_err, func_mean+n_sigma_rec*func_err, \
                                   facecolor=color, edgecolor=color, alpha=alpha, hatch=hatch, label=label)
            if key=='BaryonAcousticOscillations':
                ax[i].fill_between(x_rec, \
                                   (func_mean-n_sigma_rec*func_err)/(x_rec**(2/3)), \
                                   (func_mean+n_sigma_rec*func_err)/(x_rec**(2/3)), \
                                   facecolor=color, edgecolor=color, alpha=alpha, hatch=hatch, label=label)

        ax[-1].set_xlabel(r'Redshift $z$')

        # return fig and ax only if a new figure was created (i.e., if ax was None)
        if fig is not None:
            return fig, ax
        else:
            return None  # no return when ax is passed   

    def show_trained_ml(self, ax=None, figsize=(10, 10), method='GP', rasterized=False, \
                        n_sigma_rec=2, n_recz=100, color='red', alpha=0.25, hatch=None, label='None'):
        fig = None  # initialize fig to None
        if ax is None:
            fig, ax = plt.subplots(nrows=len(self.mock_data.keys()), figsize=figsize)
        
        for i, key in enumerate(self.mock_data.keys()):
            x_train=self.mock_data[key]['train']['x']
            if key=='SuperNovae':
                x_rec=np.linspace(np.log10(min(x_train)), np.log10(max(x_train)), n_recz)
            else:
                x_rec=np.linspace(min(x_train), max(x_train), n_recz)

            # reconstruction
            if method=='GP':
                rec_cosmo=self.GP_dict[key]
                func_mean, func_err = rec_cosmo.predict(x_rec.reshape(-1, 1), return_std=True)
            if method=='BRR':
                rec_cosmo=self.BRR_dict[key]
                _, func_mean, func_var = rec_cosmo.predict(x_rec).values()
                func_err = np.sqrt(func_var)
            if method=='ANN':
                # rec_cosmo=self.ANN_dict[key]['ANN']
                rec_cosmo=self.ANN_dict[key]
                func_rec = rec_cosmo.predict(x_rec)
                func_mean = func_rec[:, 1]; func_err = func_rec[:, 2]

            if key!='BaryonAcousticOscillations':
                if key != 'SuperNovae':
                    ax[i].fill_between(x_rec, func_mean-n_sigma_rec*func_err, func_mean+n_sigma_rec*func_err, \
                                       facecolor=color, edgecolor=color, alpha=alpha, hatch=hatch, label=label, \
                                       rasterized=rasterized)
                if key == 'SuperNovae':
                    ax[i].fill_between(10**x_rec, func_mean-n_sigma_rec*func_err, func_mean+n_sigma_rec*func_err, \
                                       facecolor=color, edgecolor=color, alpha=alpha, hatch=hatch, label=label, \
                                       rasterized=rasterized)
                    
            if key=='BaryonAcousticOscillations':
                ax[i].fill_between(x_rec, \
                                   (func_mean-n_sigma_rec*func_err)/(x_rec**(2/3)), \
                                   (func_mean+n_sigma_rec*func_err)/(x_rec**(2/3)), \
                                   facecolor=color, edgecolor=color, alpha=alpha, hatch=hatch, label=label, \
                                   rasterized=rasterized)

        ax[-1].set_xlabel(r'Redshift $z$')

        # return fig and ax only if a new figure was created (i.e., if ax was None)
        if fig is not None:
            return fig, ax
        else:
            return None  # no return when ax is passed        

