# Import Libraries 

import numpy as np

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
    

class ExtendedTrapezium(GaussianIntegrator):
    
    def integrate(func, a, b, num_points = 1000):

        x = np.linspace(a, b, num_points)  
        y = func(x)

        # Calculate step size
    
        h = (b - a) / (num_points - 1)

        # Calculate integral

        integral = 0.5 * (y[0] + y[-1]) #First and last term
        integral += sum(y[1:-1]) # Middle terms, first part is inclusive, second argument is not inclusive
        integral *= h 

        return integral
    



