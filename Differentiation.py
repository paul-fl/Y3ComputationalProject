

class FiniteDifference:
    
    def __init__(self, func, point, step_size = 1e-5):
        self.func = func
        self.point = point 
        self.step_size = step_size

    def forward_difference(self):

        point_shift_array = self.point[:]
        grad = []
        
        for i in range(len(self.point)):

            point_shift_array[i] = self.point[i] + self.step_size
            grad_i = (self.func(point_shift_array) - self.func(self.point)) / self.step_size
            grad.append(grad_i)

            point_shift_array[i] = self.point[i]

        return grad
    





        
        