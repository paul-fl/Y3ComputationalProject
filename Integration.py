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

    def __init__(self, mean = 0, sigma = 1, tol = 1e-6):
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

        steps = 2
        previous_result = 0
        while True:
            x = np.linspace(a, b, steps)
            y = func(x)
            h = (b - a) / (steps - 1)
            result = 0.5 * (y[0] + y[-1]) + sum(y[1:-1])
            result *= h

            error = abs((result - previous_result) / result)
            if error < self.tol:
                return result, steps

            previous_result = result
            steps *= 2
    
class Simpsons(GaussianIntegrator):

    def integrate(self, func, a, b, num_points = 1000):

        steps = 2
        previous_result = 0
        while True:
            trapezium = ExtendedTrapezium()
            T_j, _= trapezium.integrate(func, a, b, num_points=steps)
            T_j_next, _ = trapezium.integrate(func, a, b, num_points=2 * steps)
            result = (4 / 3) * T_j_next - (1 / 3) * T_j

            error = abs((result - previous_result) / result)
            if error < self.tol:
                return result, steps

            previous_result = result
            steps *= 2
    
class MonteCarlo(GaussianIntegrator):
     
    def integrate(self, func, a, b, max_samples=1000000):

        samples = 1
        previous_result = 0
        while samples < max_samples:
            random_samples = np.random.uniform(a, b, samples)
            func_values = func(random_samples)
            result = (b - a) * np.mean(func_values)
            #print(result)

            error = abs((result - previous_result) / result)
            if error < self.tol:
                return result, samples

            previous_result = result
            samples += 1
        
        raise RuntimeError(f"Failed to converge within {max_samples} samples.")
    
class ODE(GaussianIntegrator):
   
    def euler(self, func, a, b, max_steps = 100000):
        steps = 2  
        previous_result = 0

        while steps < max_steps:
            h = (b - a) / steps
            integral = 0
            x = a

            for _ in range(steps):
                integral += h * func(x)
                x += h
            #print(integral)

            error = abs((integral - previous_result) / integral)
            if error < self.tol:
                return integral, steps

            previous_result = integral
            steps *= 2  

        return integral, steps

    
    def rk4(self, func, a, b, max_steps = 1000000):
        steps = 2 
        previous_result = 0

        while steps < max_steps:
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

            error = abs((integral - previous_result) / integral)
            if error < self.tol:
                return integral, steps

            previous_result = integral
            steps *= 2  

        raise RuntimeError(f"RK4 method failed to converge within {max_steps} steps.")
  
class Improper(Simpsons):

    def integrate(self, func, a, b, max_steps = 10000):

        
        steps = 2
        previous_result = 0
        while steps < max_steps:
            h = 1 / steps
            midpoints = [b + (k + 0.5) * h for k in range(steps)]
            midpoint_integral = h * sum(func(m) for m in midpoints)
            result = 0.5 - midpoint_integral

            error = abs((result - previous_result) / result)
            if error < self.tol:
                return result, steps

            previous_result = result
            steps *= 2

        raise RuntimeError(f"Midpoint method failed to converge within {max_steps} steps.")

if __name__ == "__main__":

## Check Integration methods work ##

    erf_function = ErrorFunctionTest()
    trapezium  = ExtendedTrapezium()
    montecarlo = MonteCarlo()
    ode = ODE()
    improper = Improper()

    if False:

        erf_result = erf_function.integrate(a = 0, b = 5)
        trapezium_result, _ = trapezium.integrate(erf_function.gaussian, a = 0, b = 5)
        montecarlo_result, _ = montecarlo.integrate(erf_function.gaussian, a = 0, b = 5)
        euler_result, _ = ode.euler(erf_function.gaussian, a = 0,  b = 5)
        rk4_result, _ = ode.rk4(erf_function.gaussian, a = 0,  b = 5)
        improper_result, _ = improper.integrate(erf_function.gaussian, a = 0, b = 5, steps = 1000)

        print(f"Actual result: {erf_result}")
        print(f"Trapezium result: {trapezium_result}")
        print(f"MonteCarlo result: {montecarlo_result}")
        print(f"ODE Euler result: {euler_result}")
        print(f"ODE RK4 result: {rk4_result}")
        print(f"Improper result: {improper_result}")
 
    ## Comparisson of number steps to iterate, tiem taken ##
    if True:
        
        a, b = 0, 5

        exact_int = ErrorFunctionTest()
        exact_result = exact_int.integrate(a, b)
        function = exact_int.gaussian

        trapezium_iterative = ExtendedTrapeziumIterative()
        trapezium_set = ExtendedTrapezium()
        simpsons = Simpsons()
        monte_carlo = MonteCarlo()
        ode = ODE()
        improper = Improper()

        trapezium_errors = []
        simpsons_errors = []
        monte_carlo_errors = []
        euler_errors = []
        rk4_errors = []
        improper_errors = []

        b_values = np.linspace(0.1, 5, 100)

        # Trapezium
        for b in b_values:
            start_time = time.time()
            result, steps = trapezium_set.integrate(function, a, b)
            time_taken = time.time() - start_time
            trapzeium_error = abs((exact_result - result) / exact_result) * 100
            trapezium_errors.append(trapzeium_error)

        print(f"Trapezium: Final Result = {result:.10f}, Error = {trapzeium_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

        # MonteCarlo
        
        for b in b_values:
            start_time = time.time()
            result, samples = monte_carlo.integrate(function, a, b)
            time_taken = time.time() - start_time
            monte_carlo_error = abs((exact_result - result) / exact_result) * 100
            monte_carlo_errors.append(monte_carlo_error)

        print(f"MonteCarlo: Final Result = {result:.10f}, Error = {monte_carlo_error:.10f}%, Samples = {samples}, Time Taken = {time_taken:.6f}s")
        
        # Euler

        for b in b_values:
            start_time = time.time()
            result, steps = ode.euler(function, a, b, steps)
            time_taken = time.time() - start_time
            euler_error = abs((exact_result - result) / exact_result) * 100
            euler_errors.append(euler_error)

        print(f"Euler: Final Result = {result:.10f}, Error = {euler_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")
        
        # rk4
        for b in b_values:
            start_time = time.time()
            result, steps = ode.rk4(function, a, b)
            time_taken = time.time() - start_time
            rk4_error = abs((exact_result - result) / exact_result) * 100
            rk4_errors.append(rk4_error)

        print(f"RK4: Final Result = {result:.10f}, Error = {rk4_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

        # Simpsons
        for b in b_values:
            start_time = time.time()
            result, steps = simpsons.integrate(function, a, b)
            time_taken = time.time() - start_time
            simpsons_error = abs((exact_result - result) / exact_result) * 100
            simpsons_errors.append(simpsons_error)

        print(f"Simpsons: Final Result = {result:.10f}, Error = {simpsons_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

        # Improper
        for b in b_values:
            start_time = time.time()
            result, steps = improper.integrate(function, a, b)
            time_taken = time.time() - start_time
            improper_error = abs((exact_result - result) / exact_result) * 100
            improper_errors.append(improper_error)

        print(f"Midpoint: Final Result = {result:.10f}, Error = {improper_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

    ## Plot how the error changes with different upper limit ##

    plt.scatter(b_values, trapezium_errors, marker = 'x')
    plt.show()
    plt.scatter(b_values, monte_carlo_errors, marker = 'x')
    plt.show()
