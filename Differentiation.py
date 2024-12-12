import numpy as np
import time

class FiniteDifference:
    
    def __init__(self, func, point, step_size = 1e-9):
        self.func = func
        self.point = np.array(point , dtype = float)
        self.step_size = step_size

    def forward_difference(self):

        grad = np.zeros_like(self.point)
        
        
        for i in range(len(self.point)):
            
            point_array = self.point.copy()
            point_array[i] += self.step_size
            grad[i] = (self.func(point_array[0], point_array[1]) - self.func(self.point[0], self.point[1])) / self.step_size

            point_array[i] = self.point[i]

        return grad
    
    def central_difference(self):
        grad = np.zeros_like(self.point)
        
        for i in range(len(self.point)):
            
            point_forward = self.point.copy()
            point_back = self.point.copy()
            
            point_forward[i] += self.step_size
            point_back[i] -= self.step_size
            
            grad[i] = (self.func(point_forward[0], point_forward[1]) - 
                       self.func(point_back[0], point_back[1])) / (2 * self.step_size)
        
        return grad

if __name__ == "__main__":

    def gaussian(x, y):
        return np.exp(-(x**2 + y**2))

    test_gaussian = FiniteDifference(gaussian, [1.0, 2.0])

    start_time = time.time()
    FDS_result = test_gaussian.forward_difference()
    FDS_time = time.time() - start_time

    start_time = time.time()
    CDS_result = test_gaussian.central_difference()
    CDS_time = time.time() - start_time

    actual_gradient = np.array([-2 * np.exp(-5), -4 * np.exp(-5)]) 

    FDS_error = np.linalg.norm(FDS_result - actual_gradient)
    CDS_error = np.linalg.norm(CDS_result - actual_gradient)


    print(f"FDS Time {FDS_time}")
    print(f"FDS Error {CDS_error}")

    print(f"CDS Time {CDS_time}")
    print(f"CDS Error {CDS_error}")








        
        