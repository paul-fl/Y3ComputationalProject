"""
A module to perform the significance analysis for section 4
"""

# Import libraries and classes

import numpy as np
import matplotlib.pyplot as plt
from Integration import ExtendedTrapezium
from Maximisation import Maximisation


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
    
class SignificanceAnalysis:

    def __init__(self, back_func, exp_func, int_method):
        self.back_func = back_func
        self.exp_func = exp_func
        self.int_method = int_method

    def calculate_significance(self, m_l, m_u):
        NB = self.int_method.integrate(self.back_func, m_l, m_u)
        NH = self.int_method.integrate(self.exp_func, m_l, m_u)

        S = NH/np.sqrt(NB)

        return S
    

# Run the program 
## plot the significance ##

if __name__ == "__main__":
    background = BackgroundFunction()
    experimental = ExperimentalFunction()

    significance_analysis = SignificanceAnalysis(background, experimental, ExtendedTrapezium)

    m_l_values = np.linspace(120, 124, 10)
    m_u_values = np.linspace(126, 130, 10)

    sig_matrix = np.zeros((len(m_l_values), len(m_u_values)))

    for i, m_l in enumerate(m_l_values):
        for j, m_u in enumerate(m_u_values):
            if m_l < m_u:  
                S = significance_analysis.calculate_significance(m_l, m_u)
                sig_matrix[i, j] = S
            else:
                sig_matrix[i, j] = np.nan  

    plt.imshow(sig_matrix, extent=[m_u_values[0], m_u_values[-1], m_l_values[-1], m_l_values[0]],  origin='upper')
    plt.xlabel('m_u')
    plt.ylabel('m_l')
    plt.colorbar(label="Significance (S)")
    plt.title("Signifance heatmap for different levels of m_l and m_u")
    plt.show()

## calculate maximum significance ##

def significance_function(m_l, m_u):
    return significance_analysis.calculate_significance(m_l, m_u)

maximisation = Maximisation(func=significance_function, m_l_values=m_l_values, m_u_values=m_u_values)

best_values, max_s = maximisation.grid_search()

print(f"max s value: {max_s}, best values: {best_values}")


        


