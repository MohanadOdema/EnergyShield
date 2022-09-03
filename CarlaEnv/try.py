import random
import time
import math
import numpy as np
from matplotlib.pyplot import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

#===============================================================================
# True Value Samplers
#===============================================================================

class TrueSamples():
    def __init__(self, params):
        pass

    def average(self, _list, stop_index=-1):
        # average of prior values
        return sum(_list[:stop_index]) / len(_list[:stop_index]) 

class RayleighSampler(TrueSamples):
    # Data rates
    def __init__(self, params):
        self.rayleigh_sigma = params["rayleigh_sigma"]
        super().__init__(params)

    def sample(self, no_of_samples=1):
        return np.random.rayleigh(self.rayleigh_sigma, no_of_samples)

class ShiftedGammaSampler(TrueSamples):
    #  RTT delays
    def __init__(self, params):
        self.gamma_shape = params["gamma_shape"]
        self.gamma_scale = params["gamma_scale"]
        self.gamma_shift = params["gamma_shift"]
        super().__init__(params)

    def sample(self, no_of_samples=1):
        assert self.gamma_shift >= 0
        return np.random.gamma(self.gamma_shape, self.gamma_scale, no_of_samples) + self.gamma_shift

class NetworkQueueModel(TrueSamples):
    # Queuing Delays
    def __init__(self, params):
        queue_size = params['qsize']
        avg_arrival_rate = params['arate']
        avg_service_rate = params['srate']
        self.load = avg_arrival_rate/avg_service_rate
        self.xk = np.arange(queue_size)
        self.pk = [((1 - self.load) * self.load**step) / (1-self.load**(queue_size+1)) for step in self.xk]
        self.pk_sum = sum(self.pk)
        self.pk_norm = tuple(p / self.pk_sum for p in self.pk)
        self.distribution = rv_discrete(name='Queuing', values=(self.xk, self.pk_norm))
        super().__init__(params)

    def sample(self, no_of_samples=1):
        # assuming each task takes 1 ms
        occupancy = self.server_delays.distribution.rvs(size=no_of_samples)
        wait_time = occupancy/srate
        return wait_time

params = {}
params['gamma_shape'] = 3.5
params['gamma_scale'] = 5.5
params['gamma_shift'] = 46.5

n=1000000

sampler = ShiftedGammaSampler(params)
samples = sampler.sample(n)

n_bins = 500

fig, ax = plt.subplots(figsize=(10,7))
ax.hist(samples, bins=n_bins)

plt.show()







