"""
A module to develop different maximizer techniques
    
"""

# Import necessary libraries and modules 

import numpy as np
from Differentiation import FiniteDifference

class Maximisation:

    def __init__(self, func, m_l_values, m_u_values):
        
        self.func = func
        self.m_l_values = m_l_values
        self.m_u_values = m_u_values

    def grid_search(self):

        guess = None
        best_func_value = -np.inf

        for m_l in self.m_l_values:
            for m_u in self.m_u_values:
                if m_l < m_u:
                    func_value = self.func(m_l, m_u)
                    if func_value > best_func_value:
                        best_func_value = func_value
                        guess = (m_l, m_u)
        
        return guess, best_func_value

    def gradient_method(self, alpha = 1e-5, max_iterations = 1000, tolerance = 1e-6):
        pass
        