"""
A module to perform the significance analysis for section 4
"""

# Import libraries and classes

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from Functions import BackgroundFunction, ExperimentalFunction
from Integration import ExtendedTrapezium
from Maximisation import Maximisation
from Inputs import InputAnalysis



        
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
    
    def five_sigma(self, m_l, m_u):
        NB = self.int_method.integrate(self.back_func, m_l, m_u)
        NH = self.int_method.integrate(self.exp_func, m_l, m_u)
        five_sigma = 5 * np.sqrt(NB) + NB

        total = NH + NB

        p = 1 - poisson.cdf(five_sigma - 1, total)

        return p



# Run the program 
## plot the significance ##

if __name__ == "__main__":
    background = BackgroundFunction()
    experimental = ExperimentalFunction()

    significance_analysis = SignificanceAnalysis(background, experimental, ExtendedTrapezium)

    m_l_values = np.linspace(120, 124, 100)
    m_u_values = np.linspace(126, 130, 100)

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

## Calculate maximum significance ##

def significance_function(m_l, m_u):
    return significance_analysis.calculate_significance(m_l, m_u)

maximisation = Maximisation(func=significance_function, m_l_values=m_l_values, m_u_values=m_u_values)

grid_search_values, grid_search_s = maximisation.grid_search()

best_values, max_s = maximisation.grid_search()
print(f"Grid Search - Best Values: {grid_search_values}, Max Significance: {grid_search_s}")

gradient_values, gradient_s = maximisation.gradient_method(best_values)
print(f"Gradient Method - Best Values: {gradient_values}, Max Significance: {gradient_s}")

## Find probability fo significance 

m_l_best, m_u_best = gradient_values

five_sigma_prob = significance_analysis.five_sigma(m_l_best, m_u_best)
print(f"Five-Sigma Probability: {five_sigma_prob:.6f}")

## Shifting input parameters ##
input_analysis = InputAnalysis(ExtendedTrapezium)
# Shifting m_H 

shifts_MH = np.linspace(-0.2, 0.2, 100)

NH_m_H = input_analysis.shifted_MH(m_l_best, m_u_best, shifts_MH)

plt.plot(shifts_MH, NH_m_H, marker='x')
plt.xlabel("m_H shift")
plt.ylabel("Value of NH")
plt.show()

# Shifting photons

shifts_photons = np.linspace(0, 0.04, 100)

NH_photons = input_analysis.shifted_photon(m_l_best, m_u_best, shifts_photons)

plt.plot(shifts_photons, NH_photons, marker='x')
plt.xlabel("photon shift")
plt.ylabel("Value of NH")
plt.show()

# Shifting theory

shifts_theory = np.linspace(-0.03, 0.03, 100)

NH_theory = input_analysis.shifted_theory(m_l_best, m_u_best, shifts_theory)

plt.plot(shifts_theory, NH_theory, marker='x')
plt.xlabel("theory shift")
plt.ylabel("Value of NH")
plt.show()