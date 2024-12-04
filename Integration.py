# Import Libraries 

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# Define integrator class 

class GaussianIntegrator:
    """
    Base class for performing integration on a Gaussian
    """

    def __init__(self, mean = 0, sigma = 1):
        self.mean = mean
        self.sigma = sigma
    
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
    
    def integrate(func, a, b, num_points = 10):


        x = np.linspace(a, b, num_points)  
        y = func(x)
    
        h = (b - a) / (num_points - 1)

        integral = 0.5 * (y[0] + y[-1])
        integral += sum(y[1:-1]) 
        integral *= h 

        return integral
    
class MonteCarlo(GaussianIntegrator):
    pass
    
if __name__ == "__main__":
     
    erf_function = ErrorFunctionTest()
    trapezium  = ExtendedTrapezium()

    a_values = np.linspace(0, 5, 100)  
    trapezium_results = []
    erf_results = []
    trapezium_errors = []

    for a in a_values:
        erf_result = erf_function.integrate(0, a)
        trapezium_result = trapezium.integrate(0, a)

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
