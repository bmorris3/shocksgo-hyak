import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import R_sun, R_earth, M_sun
from batman import TransitParams, TransitModel

import emcee
from copy import deepcopy
from celerite import terms, GP, modeling
from multiprocessing import Pool

from shocksgo import generate_solar_fluxes

def keplers_law(period):
    return (period/365.25)**(2/3) * u.AU


rp_rs = float(R_earth/R_sun)
b = 0 

period = 5 * u.day
a_rstar = float(keplers_law(5)/R_sun)

params = TransitParams()
params.per = period.to(u.day).value
params.rp = rp_rs
params.t0 = 0 
params.inc = 90
params.w = 90 
params.ecc = 0
params.a = a_rs
params.limb_dark = 'quadratic'
params.u = [0.24, 0.36]

times = np.linspace(-2, 2, 1000)

model = TransitModel(params, times).light_curve(params)

n_trials = 25
t0_chains = []

for trial in range(n_trials):
    t, f, k = generate_solar_fluxes(times.ptp()*u.day, cadence=(times[1]-times[0])*u.day)
    fluxes = model + f

    yerr = np.std(f)
    plt.figure()
    plt.errorbar(times, fluxes, yerr, marker='.', ecolor='gray')
    plt.show()

    class MeanModel(modeling.Model): 
        parameter_names = ('t0', )

        def get_value(self, time): 
            trial_params = deepcopy(params)
            trial_params.t0 = self.t0
            return TransitModel(trial_params, time).light_curve(trial_params)


    # Set up the GP model
    kernel = terms.SHOTerm(log_S0=-25, log_omega0=5, log_Q=np.log(1/np.sqrt(2)), 
                           bounds=dict(log_S0=(-35, -20), log_omega0=(0, 10)))
    kernel.freeze_parameter('log_Q')
    mean_model = MeanModel(t0=0, bounds=dict(t0=(-0.1, 0.1)))
    gp = GP(kernel, mean=mean_model, fit_mean=True)
    gp.compute(times, yerr)

    def log_probability(params):
        gp.set_parameter_vector(params)
        lp = gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(fluxes) + lp


    initial = np.array([-28, 6.5, 0])
    ndim, nwalkers = len(initial), len(initial) * 2

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

        print("Running burn-in...")
        p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
        p0 = sampler.run_mcmc(p0, 2000)

        print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0, 5000);

    corner(sampler.flatchain, labels=gp.get_parameter_names());

    t0_chains.append(sampler.flatchain[:, -1])

#std = np.median([s.std() for s in t0_chains])
#error = np.std([s.std() for s in t0_chains])

np.savetxt('~/git/shocksgo-hyak/results/{0}_{1}_{2:04d}.txt'.format(period.to(u.day).value, a_rs, np.random.randint(0, 2048)), [s.std() for s in t0_chains])