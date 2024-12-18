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
import time



        
class SignificanceAnalysis:

    def __init__(self, back_func, exp_func, int_method):
        self.back_func = back_func
        self.exp_func = exp_func
        self.int_method = int_method

    def calculate_significance(self, m_l, m_u):
        NB, NB_steps = self.int_method.integrate(self.back_func, m_l, m_u)
        NH, NH_steps  = self.int_method.integrate(self.exp_func, m_l, m_u)

        S = NH/np.sqrt(NB)

        return S
    
    def five_sigma(self, m_l, m_u, combined_uncertainty):
        NB, NB_steps = self.int_method.integrate(self.back_func, m_l, m_u)
        NH, NH_steps = self.int_method.integrate(self.exp_func, m_l, m_u)
        five_sigma = 5 * np.sqrt(NB + combined_uncertainty) + NB 

        total = NH + NB

        p = 1 - poisson.cdf(five_sigma - 1, total)

        return p
    

# Run the program 
## plot the significance ##

if __name__ == "__main__":
    background = BackgroundFunction()
    experimental = ExperimentalFunction()
    trapezium = ExtendedTrapezium(tol = 1e-6)

    significance_analysis = SignificanceAnalysis(background, experimental, trapezium)

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
    print(f"Grid Search Values: {grid_search_values}, Max Sig: {grid_search_s}")

    gradient_values, gradient_s = maximisation.gradient_method(best_values)
    print(f"Gradient Method sValues: {gradient_values}, Max Sig: {gradient_s}")

    ## Calculate the maximum significance given by grid seach for differnt grid spacings ##

    if True:
        grid_spacings = np.linspace(5, 100, 100)
        sig_matrix
        max_sig = []
        time_passed = []

        for spacing in grid_spacings:
            m_l_values = np.linspace(120, 124, int(spacing))
            m_u_values = np.linspace(126, 130, int(spacing))

            def significance_function(m_l, m_u):
                return significance_analysis.calculate_significance(m_l, m_u)

            maximisation = Maximisation(func=significance_function, m_l_values=m_l_values, m_u_values=m_u_values)
            start_time = time.time()
            _, max_s = maximisation.grid_search()
            time_taken = time.time() - start_time
            time_passed.append(time_taken)
            max_sig.append(max_s)

        
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Grid Spacing')
        ax1.set_ylabel('Maximum Significance', color=color)
        ax1.scatter(grid_spacings, max_sig, marker='x', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Grid Search Time (s)', color=color)
        ax2.plot(grid_spacings, time_passed, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title("Maximum Significance and Grid Search Time vs Grid Spacing")
        plt.show()

    ## Find probability fo significance 

    m_l_best, m_u_best = gradient_values

    five_sigma_prob = significance_analysis.five_sigma(m_l_best, m_u_best, 0)
    print(f"Five-Sigma Probability: {five_sigma_prob:.6f}")

    ## Shifting input parameters ##

    input_analysis = InputAnalysis(trapezium)
    # Shifting m_H 

    shifts_MH = np.linspace(-0.2, 0.2, 100)

    NH_m_H = input_analysis.shifted_MH(m_l_best, m_u_best, shifts_MH)

    plt.plot(shifts_MH, NH_m_H, marker='x')
    plt.xlabel("m_H shift")
    plt.ylabel("Value of NH")
    plt.title("Value of NH for different shifts in m_H")
    plt.show()

    # Shifting photons

    shifts_photons = np.linspace(0, 0.04, 100)

    NH_photons = input_analysis.shifted_photon(m_l_best, m_u_best, shifts_photons)

    plt.plot(shifts_photons, NH_photons, marker='x')
    plt.xlabel("photon shift")
    plt.ylabel("Value of NH")
    plt.title("Value of NH for different shifts in photon pairs")
    plt.show()

    # Shifting theory

    shifts_theory = np.linspace(-0.03, 0.03, 100)

    NH_theory = input_analysis.shifted_theory(m_l_best, m_u_best, shifts_theory)

    plt.plot(shifts_theory, NH_theory, marker='x')
    plt.xlabel("theory shift")
    plt.ylabel("Value of NH")
    plt.title("Value of NH for different shifts in theory uncertainty")
    plt.show()

    ## Compare with the background statistical uncertainty ##

    N_B = input_analysis.calculate_NH(background, m_l_best, m_u_best)
    statistical_error = np.sqrt(N_B)

    normal_NH = input_analysis.calculate_NH(experimental, m_l_best, m_u_best)

    mass_error = max(abs(nh - normal_NH) for nh in NH_m_H)
    photons_error = max(abs(nh - normal_NH) for nh in NH_photons)
    theory_error = max(abs(nh - normal_NH) for nh in NH_theory)

    print(f"Statistical Error (Poisson): {statistical_error:.4f}")
    print(f"Max Deviation (m_H shift): {mass_error:.4f}")
    print(f"Max Deviation (Photon shift): {photons_error:.4f}")
    print(f"Max Deviation (Theory shift): {theory_error:.4f}")

    combined_unc = np.sqrt(mass_error**2 + photons_error**2 + theory_error**2)
    print(f"Total uncertainty: {combined_unc}")

    five_sigma_prob_unc = significance_analysis.five_sigma(m_l_best, m_u_best, combined_unc)
    
    print(f"Five-Sigma Probability with uncertainty: {five_sigma_prob_unc}")