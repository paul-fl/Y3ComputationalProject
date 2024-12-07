# Import Libraries 

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import time

# Define integrator class 

class GaussianIntegrator:
    """
    Base class for performing integration on a Gaussian
    """

    def __init__(self, mean = 0, sigma = 1, tol = 1e-10):
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

class ExtendedTrapeziumIterative(GaussianIntegrator):
    
    def integrate(self, func, a, b, h = 0.5, max_iter=10000):

        h = b - a  
        I_prev = h * 0.5 * (func(a) + func(b))  
        iteration = 0

        while iteration < max_iter:
            iteration += 1

            mid_points = np.linspace(a + h / 2, b - h / 2, 2**(iteration - 1))
            midpoint_sum = np.sum(func(mid_points))

            I_curr = 0.5 * I_prev + h * 0.5 * midpoint_sum

            relative_error = abs((I_curr - I_prev) / I_curr)

            if relative_error < self.tol:
                
                return I_curr

            I_prev = I_curr
            h /= 2  

        raise RuntimeError(f"Failed to converge within {max_iter} iterations.")
    
class ExtendedTrapezium(GaussianIntegrator):

    def integrate(self, func, a, b, num_points = 1000):

        x = np.linspace(a, b, num_points)  
        
        y = func(x)

        h = (b - a) / (num_points - 1)

        integral = 0.5 * (y[0] + y[-1])
        integral += sum(y[1:-1]) 
        integral *= h 

        return integral
    
class MonteCarlo(GaussianIntegrator):
     
    def integrate(self, func, a, b, num_samples=100):

        random_samples = np.random.uniform(a, b, num_samples)
    
        function_values = func(random_samples)
        
        integral_estimate = (b - a) * np.mean(function_values)
        
        return integral_estimate
    
class ODE(GaussianIntegrator):
    
    def integrate(self, a, b):
        pass
        
        

    
if __name__ == "__main__":
     
## Compare integration methods using a fixed h ##

    erf_function = ErrorFunctionTest()
    trapezium  = ExtendedTrapezium()
    montecarlo = MonteCarlo()

    a_values = np.linspace(0, 5, 500)  

    trapezium_results = []
    trapezium_errors = []
    trapezium_times = []

    montecarlo_results = []
    montecarlo_erros = []

    erf_results = []

    
    

    for a_value in a_values:
        
        erf_result = erf_function.integrate(a = 0, b = a_value)
       
        trapezium_result = trapezium.integrate(erf_function.gaussian, a = 0, b = a_value)

        montecarlo_result = montecarlo.integrate(erf_function.gaussian, a = 0, b = a_value)

        erf_results.append(erf_result)
        trapezium_results.append(trapezium_result)
        montecarlo_results.append(montecarlo_result)

        trapezium_errors.append(abs(erf_result - trapezium_result))
        montecarlo_erros.append(abs(erf_result - montecarlo_result))


    plt.scatter(a_values, trapezium_errors, label="Trapezium Error", marker="o")
    #plt.scatter(a_values, montecarlo_erros, label="MonteCarlo Error", marker="x")
    plt.xlabel("value of uppter limit")
    plt.ylabel("Absolute Error")
    plt.title("Integration Method Errors (Relative to ErrorFunction)")
    plt.legend()
    plt.grid()
    plt.show()

    ## PLOT the time it takes vs the error for each method ##

    tolerances = np.logspace(-8, -2, 1000)
    trapezium_errors = []
    trapezium_times = []

    for tol in tolerances:
        extended_trapezium = ExtendedTrapeziumIterative(tol = tol)

        start_time = time.time()
        trapezium_result = extended_trapezium.integrate(erf_function.gaussian, a=1, b=5)
        trapezium_time = time.time() - start_time
        trapezium_times.append(trapezium_time)

        erf_result = erf_function.integrate(a=1 , b=5)
        error = abs(erf_result - trapezium_result)
        trapezium_errors.append(error)

    plt.loglog(trapezium_times, trapezium_errors, label="Trapezium", marker="x")
    plt.show()

    

