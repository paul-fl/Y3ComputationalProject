import numpy as np


class BackgroundFunction:
    
    def __init__(self, A = 1500, k = 20, m_H = 125.1):

        self.A = A
        self.k = k
        self.m_H = m_H


    def __call__(self, m):

        return self.A * np.exp(-(m - self.m_H) / self.k)
    
class ExperimentalFunction:

    def __init__(self, photon_pairs=470, mean=125.1, sigma=1.4):
        self.photon_pairs = photon_pairs
        self.mean = mean
        self.sigma = sigma

    def __call__(self, m):

        return self.photon_pairs * (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((m - self.mean) ** 2) / (2 * self.sigma ** 2))