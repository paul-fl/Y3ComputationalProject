import numpy as np
import matplotlib.pyplot as plt

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
    

if __name__ == "__main__":
    background = BackgroundFunction()
    experimental = ExperimentalFunction()

    m_values = np.linspace(100, 150, 500)
    background_values = [background(m) for m in m_values]
    experimental_values = [experimental(m) for m in m_values]
    
    plt.plot(m_values, np.array(background_values) + np.array(experimental_values), label='Superimposed Function')
    plt.show()