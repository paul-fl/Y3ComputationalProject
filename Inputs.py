
import numpy as np
from Functions import ExperimentalFunction

class InputAnalysis:

    def __init__(self, int_method ,m_H = 125.1, pairs=470, pairs_unc=0.04, theory_unc=0.03):
       self.m_H = m_H
       self.pairs = pairs 
       self.pairs_unc = pairs_unc
       self.theory_unc = theory_unc
       self.int_method = int_method

    def calculate_NH(self, experimental_function, m_l, m_u):
        N_H, NH_steps = self.int_method.integrate(func = experimental_function, a = m_l, b= m_u)

        return N_H
    
    def shifted_MH(self, m_l, m_u, shifts):

        NH_values = []

        for shift in shifts:
            experimental = ExperimentalFunction(photon_pairs=470, mean=125.1 + shift, sigma=1.4)

            NH_value = self.calculate_NH(experimental, m_l, m_u)
            NH_values.append(NH_value)

        return NH_values
    
    def shifted_photon(self, m_l, m_u, fractions):
  
        NH_values = []

        for fraction in fractions:

            total_pairs = self.pairs

            shifted_pairs = (total_pairs * fraction)
            unshifted_pairs = total_pairs - shifted_pairs

            unshifted_experimental = ExperimentalFunction(photon_pairs=unshifted_pairs, mean=125.1, sigma=1.4)

            shifted_experimental = ExperimentalFunction(photon_pairs=shifted_pairs, mean=124.5, sigma=2.6)

            def combined_function(m):
                return unshifted_experimental(m) + shifted_experimental(m)

    
            NH_value = self.calculate_NH(combined_function, m_l, m_u)
            NH_values.append(NH_value)

        return NH_values
    
    def shifted_theory(self, m_l, m_u, shifts):

        NH_values = []

        for shift in shifts:
            
            actual_pairs = self.pairs * (1 + shift)

            experimental = ExperimentalFunction(photon_pairs=actual_pairs, mean=self.m_H, sigma=1.4)

            NH_value = self.calculate_NH(experimental, m_l, m_u)
            NH_values.append(NH_value)

        return NH_values


    