import numpy as np

class FiniteDifference:
    
    def __init__(self, func, point, step_size = 1e-5):
        self.func = func
        self.point = np.array(point , dtype = float)
        self.step_size = step_size

    def forward_difference(self):

        grad = np.zeros_like(self.point)
        
        
        for i in range(len(self.point)):
            
            point_shift_array = self.point.copy()
            point_shift_array[i] += self.step_size
            grad[i] = (self.func(point_shift_array[0], point_shift_array[1]) - self.func(self.point[0], self.point[1])) / self.step_size

            point_shift_array[i] = self.point[i]

        return grad
    





        
        