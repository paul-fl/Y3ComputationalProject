# Import Libraries 

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# Define integrator class 

class GaussianIntegrator:
    """
    Base class for performing integration on a Gaussian
    """

    def __init__(self, mean = 0, sigma = 1, tol = 1e-3):
        self.mean = mean
        self.sigma = sigma
        self.tol = tol
    
    def gaussian(self, x):
        
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mean) ** 2) / (2 * self.sigma ** 2))
    
    def integrate(self, a, b):
        raise NotImplementedError('integrate needs to be implemented in derived classes')

class ErrorFunctionTest(GaussianIntegrator):
        
    def integrate(self, a, b):

        z_a = a / (self.sigma * np.sqrt(2))
        z_b = b / (self.sigma * np.sqrt(2))
        
        return 0.5 * (erf(z_b) - erf(z_a))

class ExtendedTrapezium(GaussianIntegrator):
    
    def integrate(self, func, a, b, h = 0.5, max_iter=10000):

        h = b - a  # Initial step size
        I_prev = h * 0.5 * (func(a) + func(b))  # First estimate I1
        iteration = 0

        while iteration < max_iter:
            iteration += 1

            mid_points = np.linspace(a + h / 2, b - h / 2, 2**(iteration - 1))
            midpoint_sum = np.sum(func(mid_points))

            I_curr = 0.5 * I_prev + h * 0.5 * midpoint_sum

            relative_error = abs((I_curr - I_prev) / I_curr)

            if relative_error < self.tol:
                print(f"Iterated in {iteration}")
                return I_curr

            I_prev = I_curr
            h /= 2  

        raise RuntimeError(f"Failed to converge within {max_iter} iterations.")
    
class MonteCarlo(GaussianIntegrator):
    pass
    
if __name__ == "__main__":
     
## Compare integration methods using a fixed h ##

    erf_function = ErrorFunctionTest()
    trapezium  = ExtendedTrapezium()

    a_values = np.linspace(0, 5, 100)  
    trapezium_results = []
    erf_results = []
    trapezium_errors = []

    for a in a_values:
        erf_result = erf_function.integrate(0, a)
        trapezium_result = trapezium.integrate(0, a, h = 0.5)

        erf_results.append(erf_result)
        trapezium_results.append(trapezium_result)

        trapezium_errors.append(abs(trapezium_result - erf_result) / abs(erf_result))

    plt.plot(a_values, trapezium_errors, label="Trapezium Error", marker="o")
    plt.xlabel("a")
    plt.ylabel("Absolute Error")
    plt.title("Integration Method Errors (Relative to ErrorFunction)")
    plt.legend()
    plt.grid()
    plt.show()
