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
   
    def euler(self, func, a, b, max_steps = 1000000):
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

    def integrate(self, func, b, max_steps=10000):
        delta = 1.0  
        simpson_integrator = Simpsons(tol=self.tol)
        simpson_result, _ = simpson_integrator.integrate(func, b, b + delta)

        steps = 2
        h = 1.0 / steps
        previous_result = 0
        while steps < max_steps:
            midpoints = [b + delta + (k + 0.5) * h for k in range(steps)]
            midpoint_integral = h * sum(func(m) for m in midpoints)
            result = simpson_result + midpoint_integral 

            error = abs((result - previous_result) / result)
            if error < self.tol:
                return (0.5 - result), steps

            previous_result = result
            steps *= 2
            h = 1.0 / steps  

        raise RuntimeError(f"Failed to converge within {max_steps} steps.")

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
    a, b = 0.1, 5
   
    def linear(x):
        return x
        
    def logarithmic(x):
        return (x * np.log(x))

    exact_int = ErrorFunctionTest()
        
    exact_result = exact_int.integrate(a, b)
    
    
    
    functions = {
        "Gaussian": {
            "function": exact_int.gaussian,
            "exact_integral": exact_int.integrate  
        },
        "Linear": {
            "function": linear,
            "exact_integral": lambda a, b: (b**2 / 2) - (a**2 / 2)
        },
        "Logarithmic": {
            "function": logarithmic,
            "exact_integral": lambda a, b: ((b**2 / 2) * np.log(b) - (b**2 / 4)) - ((a**2 / 2) * np.log(a) - (a**2 / 4))
        }
    }

    for func_name, func_data in functions.items():
        function = func_data["function"]
        exact_integral = func_data["exact_integral"]
        




        #function = exact_int.gaussian
        #function = linear
        #function = logarithmic

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

        b_values = np.linspace(0.2, 5, 10)


        # Trapezium
        for b in b_values:
            start_time = time.time()
            result, steps = trapezium_set.integrate(function, a, b)
            time_taken = time.time() - start_time
            #exact_result = (b**2 / 2) - (a**2 / 2)
            #exact_result = ((b**2 / 2) * np.log(b) - (b**2 / 4)) - ((a**2 / 2) * np.log(a) - (a**2 / 4))
            exact_result = exact_int.integrate(a, b)
            trapzeium_error = abs((exact_result - result) / exact_result) * 100
            trapezium_errors.append(trapzeium_error)

        print(f"Trapezium: Final Result = {result:.10f}, Error = {trapzeium_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

        # MonteCarlo
        
        for b in b_values:
            start_time = time.time()
            result, samples = monte_carlo.integrate(function, a, b)
            time_taken = time.time() - start_time
            #exact_result = (b**2 / 2) - (a**2 / 2)
            #exact_result = ((b**2 / 2) * np.log(b) - (b**2 / 4)) - ((a**2 / 2) * np.log(a) - (a**2 / 4))
            exact_result = exact_int.integrate(a, b)
            monte_carlo_error = abs((exact_result - result) / exact_result) * 100
            monte_carlo_errors.append(monte_carlo_error)

        print(f"MonteCarlo: Final Result = {result:.10f}, Error = {monte_carlo_error:.10f}%, Samples = {samples}, Time Taken = {time_taken:.6f}s")
        
        # Euler

        for b in b_values:
            start_time = time.time()
            result, steps = ode.euler(function, a, b)
            time_taken = time.time() - start_time
            #exact_result = (b**2 / 2) - (a**2 / 2)
            #exact_result = ((b**2 / 2) * np.log(b) - (b**2 / 4)) - ((a**2 / 2) * np.log(a) - (a**2 / 4))
            exact_result = exact_int.integrate(a, b)
            euler_error = abs((exact_result - result) / exact_result) * 100
            euler_errors.append(euler_error)

        print(f"Euler: Final Result = {result:.10f}, Error = {euler_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")
        
        # rk4
        for b in b_values:
            start_time = time.time()
            result, steps = ode.rk4(function, a, b)
            time_taken = time.time() - start_time
            #exact_result = (b**2 / 2) - (a**2 / 2)
            #exact_result = ((b**2 / 2) * np.log(b) - (b**2 / 4)) - ((a**2 / 2) * np.log(a) - (a**2 / 4))
            exact_result = exact_int.integrate(a, b)
            rk4_error = abs((exact_result - result) / exact_result) * 100
            rk4_errors.append(rk4_error)

        print(f"RK4: Final Result = {result:.10f}, Error = {rk4_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

        # Simpsons
        for b in b_values:
            start_time = time.time()
            result, steps = simpsons.integrate(function, a, b)
            time_taken = time.time() - start_time
            #exact_result = (b**2 / 2) - (a**2 / 2)
            #exact_result = ((b**2 / 2) * np.log(b) - (b**2 / 4)) - ((a**2 / 2) * np.log(a) - (a**2 / 4))
            exact_result = exact_int.integrate(a, b)
            simpsons_error = abs((exact_result - result) / exact_result) * 100
            simpsons_errors.append(simpsons_error)

        print(f"Simpsons: Final Result = {result:.10f}, Error = {simpsons_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

        # Improper
        for b in b_values:
            start_time = time.time()
            result, steps = improper.integrate(function, b)
            time_taken = time.time() - start_time
            #exact_result = (b**2 / 2) - (a**2 / 2)
            #exact_result = ((b**2 / 2) * np.log(b) - (b**2 / 4)) - ((a**2 / 2) * np.log(a) - (a**2 / 4))
            exact_result = exact_int.integrate(a, b)
            improper_error = abs((exact_result - result) / exact_result) * 100
            improper_errors.append(improper_error)

        print(f"Midpoint: Final Result = {result:.10f}, Error = {improper_error:.10f}%, Steps = {steps}, Time Taken = {time_taken:.6f}s")

    ## Plot how the error changes with different upper limit ##
    title_fontsize = 30
    label_fontsize = 24
    tick_fontsize = 20

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(b_values, simpsons_errors, label="Simpson's", marker='x')
    axs[0].set_title("Simpson's Method", fontsize=title_fontsize)
    axs[0].set_xlabel("Upper Limit of Integration (a)", fontsize=label_fontsize)
    axs[0].set_ylabel("Relative Error (%)", fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].grid()

    axs[1].plot(b_values, trapezium_errors, label="Trapezium", marker='x')
    axs[1].set_title("Trapezium Method", fontsize=title_fontsize)
    axs[1].set_xlabel("Upper Limit of Integration (a)", fontsize=label_fontsize)
    axs[1].set_ylabel("Relative Error (%)", fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(b_values, monte_carlo_errors, label="Monte Carlo", marker='s')
    axs[0].set_title("Monte Carlo Method", fontsize=title_fontsize)
    axs[0].set_xlabel("Upper Limit of Integration (a)", fontsize=label_fontsize)
    axs[0].set_ylabel("Relative Error (%)", fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].grid()

    axs[1].plot(b_values, improper_errors, label="Composite", marker='x')
    axs[1].set_title("Composite Method", fontsize=title_fontsize)
    axs[1].set_xlabel("Upper Limit of Integration (a)", fontsize=label_fontsize)
    axs[1].set_ylabel("Relative Error (%)", fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(b_values, euler_errors, label="Euler", marker='x')
    axs[0].set_title("Euler Method", fontsize=title_fontsize)
    axs[0].set_xlabel("Upper Limit of Integration (a)", fontsize=label_fontsize)
    axs[0].set_ylabel("Relative Error (%)", fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].grid()

    axs[1].plot(b_values, rk4_errors, label="RK4", marker='x')
    axs[1].set_title("RK4 Method", fontsize=title_fontsize)
    axs[1].set_xlabel("Upper Limit of Integration (a)", fontsize=label_fontsize)
    axs[1].set_ylabel("Relative Error (%)", fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].grid()

    plt.tight_layout()
    plt.show()