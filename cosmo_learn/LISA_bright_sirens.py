#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:53:00 2024

@author: genebelinario
"""

# imports
from math import pi, sqrt, atan, floor
from scipy.misc import derivative
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
# import random
# from random import uniform, gauss
import numpy as np
from astropy.cosmology import w0waCDM

# import cosmo.py
#from .cosmo import H, dL, GetRandom, dL_line, distribute

#####

# define Hubble function
def H(z, params):
    
    # specifcy params with the parameters H0, Om0, Ok0, w0, wa in order
    H0_apy = params[0]
    Om0_apy = params[1]
    Ok0_apy = params[2]
    w0_apy = params[3]
    wa_apy = params[4]
    
    h = H0_apy/100
    H0 = 299792458*h/(2.9979*3.085678*10**25)
    
    cosmo = w0waCDM(H0=H0_apy, Om0=Om0_apy, Ode0=1-Om0_apy-Ok0_apy, w0=w0_apy, wa=wa_apy, Tcmb0=2.725)
    
    Hz_apy = cosmo.H(z).value
    Ez_apy = Hz_apy/H0_apy
    
    return H0*Ez_apy

#####

# define the luminosity distance function
def dL(z, H, params):
    c = 9.715611890800001e-18  # speed of light [Gpc/s]
    return (1+z) * c * quad(lambda Z: 1/H(Z, params), 0, z)[0] 

#####

# get N randomly generated events from a given distribution, using rejection methods
def GetRandom(distribution, x_min, x_max, y_min, y_max, N=1):
    counter = 0
    events = []

    while counter < N:
        # x = uniform(x_min, x_max)
        # y = uniform(y_min, y_max)
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        if y < distribution(x):
            events.append(x)
            counter += 1

    return events

#####

# theoretical line for luminosity distance
def dL_line(zmin, zmax, N=1000):
    # protection against invalid arguments
    if (zmin < 0 or zmax < 0) or (zmax < zmin):
        raise Exception("Please specify a valid redshifts interval.")

    # create a "solid line" and compute distances for that line
    line = np.linspace(zmin, zmax, N)
    distances = [dL(i, H) for i in line]

    return line, distances

#####

# convert luminosity distance to redshift
def dL_to_redshift(distance, z0=0):
    # auxiliary function to solve using scipy
    def func(z, distance, H):
        return distance - dL(z, H)

    # compute the redshift for the provided luminosity distance
    redshift = fsolve(func, z0, args=(distance, H))[0]

    return redshift

#####

# distribute the events around the most likely value using a gaussian distribution, with protection against negative values
def distribute(distances, errors):
    for i in range(0, len(distances)):
        newdistance = -1
        while newdistance < 0:
            # newdistance = gauss(distances[i], errors[i])
            newdistance = np.random.normal(loc=distances[i], scale=errors[i])
        distances[i] = newdistance

    return distances, errors

#####

def dist(population):
    # redshift boundaries for LISA
    zmin = 0.1
    zmax = 9

    # probability distribution for the provided population
    if population == "Pop III":
        dist = [2.012, 7.002, 8.169, 5.412, 3.300, 1.590, 0.624, 0.141, 0.000]

    elif population == "Delay":
        dist = [0.926, 4.085, 5.976, 5.131, 4.769, 2.656, 1.710, 0.644, 0.362]

    elif population == "No Delay":
        dist = [3.682, 10.28, 9.316, 7.646, 4.909, 2.817, 1.187, 0.362, 0.161]

    # get the total number of events
    N = dist[0]*0.9 + sum(dist[1:])

    # normalize the distribution
    dist = [i/N for i in dist]

    # get the minimum and maximum of the redshift distribution
    dmin = 0
    dmax = max(dist)

    # define our redshift distribution function
    def f(z):
        if z < 0.1 or z >= 9:
            return 0
        return dist[floor(z)]

    return (f, zmin, zmax, dmin, dmax, N)

#####

# errors for the luminosity distance
# from arXiv:2010.09049, page 6
def sigma_lens(z, dL, H, params):
    return 0.066 * ((1-(1+z)**(-0.25))/0.25)**(1.8) * dL(z, H, params)

def F_delens(z):
    return 1 - 2*0.3/pi * atan(z/0.073)

def sigma_delens(z, dL, H, params):
    return sigma_lens(z, dL, H, params) * F_delens(z)

def sigma_v(z, dL, H, params):
    rms = 1.6203896*10**(-20)   # [Gpc/s]
    c = 9.7156118908*10**(-18)  # speed of light [Gpc/s]
    return ( ( 1 + (c*(1+z)**2)/(H(z, params)*dL(z, H, params)) ) * rms/c ) * dL(z, H, params)

def sigma_LISA(z, dL, H, params):
    return 0.05 * (dL(z, H, params)**2)/36.6

def sigma_photo(z):
    if z < 2:
        return 0
    return 0.03*(1+z)

def error(z, dL, H, params):
    return sqrt(sigma_delens(z, dL, H, params)**2 + sigma_v(z, dL, H, params)**2 + sigma_LISA(z, dL, H, params)**2 + (derivative(dL, z, dx=1e-6, args=(H,params,)) * sigma_photo(z))**2)

#####

# generate the forecast LISA events
def generate(population="No Delay", events=0, years=0, redshifts=[], ideal=False, params=[]): # , seed=None):
    # if seed is not None:
    #     # np.random.seed(seed)
    #     random.seed(seed)

    # protection against none or invalid population
    if not population:
        raise Exception("The population of MBHB must be provided, available populations are: 'Pop III', 'Delay' and 'No Delay'")
    if population not in ["Pop III", "Delay", "No Delay"]:
        raise Exception("Population not available, available populations are: 'Pop III', 'Delay' and 'No Delay'")

    # specify either events, years or redshifts
    if bool(events) + bool(years) + bool(redshifts) != 1:
        raise Exception("Specify either the number of events, years or redshifts")

    # get the redshift distribution function, minimums/maximums and number of events for that distribution
    f, zmin, zmax, dmin, dmax, N = dist(population)

    # get luminosity distance and error for specific redshifts
    if redshifts:
        # protect against out of bound redshifts
        if min(redshifts) < zmin or max(redshifts) > zmax:
            raise Exception(f"Redshift limits are out of bounds. Lowest and highest redshift for LISA are z={zmin} and z={zmax} correspondingly")

        distances = [dL(z, H) for z in redshifts]
        errors = [error(z, dL, H) for z in redshifts]

    # generate events according to the redshift distribution
    else:
        if events != 0:
            N = events
        elif years != 0:
            N = int(N * years/5)

        # get redshifts and the distance and error for each event
        redshifts = GetRandom(f, zmin, zmax, dmin, dmax, N=N)
        distances = [dL(z, H, params) for z in redshifts]
        errors = [error(z, dL, H, params) for z in redshifts]

    # distribute the events around the most likely value using a gaussian distribution
    if not ideal:
        distances, errors = distribute(distances, errors)

    return redshifts, distances, errors