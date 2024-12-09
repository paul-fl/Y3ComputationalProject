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
    
class Simpsons(GaussianIntegrator):

    def integrate(self, func, a, b, num_points = 1000):
        
        trapezium = ExtendedTrapezium()
        T_j = trapezium.integrate(func, a, b, num_points=num_points)

        T_j_next = trapezium.integrate(func, a, b, num_points=2 * num_points)

        S_j = (4 / 3) * T_j_next - (1 / 3) * T_j
        return S_j
    
class MonteCarlo(GaussianIntegrator):
     
    def integrate(self, func, a, b, samples=10000):

        random_samples = np.random.uniform(a, b, samples)
    
        func_values = func(random_samples)
        
        integral = (b - a) * np.mean(func_values)
        
        return integral
    
class ODE(GaussianIntegrator):
   
    def euler(self, func, a, b, steps = 1000):
        h = (b - a) / steps
        integral = 0  
        x = a

        for _ in range(steps):
            integral += h * func(x) 
            x += h  

        return integral
    
    def rk4(self, func, a, b, steps = 1000):
        h = (b - a) / steps
        integral = 0  
        x = a

        for _ in range(steps):
            k1 = func(x)
            k2 = func(x + h / 2)
            k3 = func(x + h / 2)
            k4 = func(x + h)

            integral += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x += h  
        
        return integral
    
if __name__ == "__main__":

## Check Integration methods work ##

    erf_function = ErrorFunctionTest()
    trapezium  = ExtendedTrapezium()
    montecarlo = MonteCarlo()
    ode = ODE()

    erf_result = erf_function.integrate(a = 0, b = 5)
    trapezium_result = trapezium.integrate(erf_function.gaussian, a = 0, b = 5)
    montecarlo_result = montecarlo.integrate(erf_function.gaussian, a = 0, b = 5)
    euler_result = ode.euler(erf_function.gaussian, a = 0,  b = 5)
    rk4_result = ode.rk4(erf_function.gaussian, a = 0,  b = 5)

    print(f"Actual result: {erf_result}")
    print(f"Trapezium result: {trapezium_result}")
    print(f"MonteCarlo result: {montecarlo_result}")
    print(f"ODE Euler result: {euler_result}")
    print(f"ODE RK4 result: {rk4_result}")

     
## Compare integration methods on varying upper limit#

    if False:
    
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

    ## Final Comparission ##

    a, b = 0, 5

    exact_int = ErrorFunctionTest()
    exact_result = exact_int.integrate(a, b)
    func = exact_int.gaussian

    tolerance = 1e-9

    trapezium_iterative = ExtendedTrapeziumIterative()
    trapezium_set = ExtendedTrapezium()
    simsons = Simpsons()
    monte_carlo = MonteCarlo()
    ode = ODE()

    # Trapezium
    steps = 2
    start_time = time.time()
    previous_result = 0 
    while True:
       
        result = trapezium_set.integrate(func, a, b, steps)
        
        error_exact = abs((exact_result - result) / exact_result) * 100
        error = abs(result - previous_result) / abs(result)

        previous_result = result

        if error < tolerance * 100:
            break
        steps += 1
    
    time_taken = time.time() - start_time
    print(f"Trapezium: Final Result = {result:.10f}, Error = {error_exact:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

    # MonteCarlo
    samples = 1
    max_samples = 10000
    start_time = time.time()
    previous_result = 0
    while samples < max_samples:
       
        result = monte_carlo.integrate(func, a, b, samples)
        error_exact = abs((exact_result - result) / exact_result) * 100
        error = abs(result - previous_result) / abs(result)

        previous_result = result

        if error < tolerance * 100:
            break
        samples += 1

    time_taken = time.time() - start_time
    print(f"MonteCarlo: Final Result = {result:.10f}, Error = {error:.10f}%, Samples = {samples}, Time Taken = {time_taken:.6f}s")
    
    # Euler
    steps = 2
    start_time = time.time()
    previous_result = 0
    while True:
        start_time = time.time()
        result = ode.euler(func, a, b, steps)
        time_taken = time.time() - start_time
        error_exact = abs((exact_result - result) / exact_result) * 100
        error = abs(result - previous_result) / abs(result)

        previous_result = result

        if error < tolerance * 100:
            break
        steps += 1
    
    time_taken = time.time() - start_time
    print(f"Euler: Final Result = {result:.10f}, Error = {error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")
    
    # rk4
    start_time = time.time()
    previous_result = 0
    steps = 2
    while True:
        start_time = time.time()
        result = ode.rk4(func, a, b, steps)
        error_exact = abs((exact_result - result) / exact_result) * 100
        error = abs(result - previous_result) / abs(result)

        previous_result = result
        if error < tolerance * 100:
            break
        steps += 1
     
    time_taken = time.time() - start_time
    print(f"RK4: Final Result = {result:.10f}, Error = {error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

    # Simpsons
    steps = 2
    start_time = time.time()
    previous_result = 0 
    while True:
       
        result = simsons.integrate(func, a, b, steps)
        
        error_exact = abs((exact_result - result) / exact_result) * 100
        error = abs(result - previous_result) / abs(result)

        previous_result = result

        if error < tolerance * 100:
            break
        steps += 1
    
    time_taken = time.time() - start_time
    print(f"Simpsons: Final Result = {result:.10f}, Error = {error_exact:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")