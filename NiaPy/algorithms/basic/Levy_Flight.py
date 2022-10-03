import numpy as np
import scipy
import math
import random as rnd

__all__ = ['Levy_Flight']

class Levy_Flight:
    
    def __init__(self):

        self.Beta = 0
        self.mu = 0
        self.sigma_u = 1
        self.sigma_v = 1

    def update_sigma_u(self):

        # randomly update Beta in the range of (0,2]
        self.Beta = rnd.uniform(0.1,2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        self.sigma_u = (math.gamma(1 + self.Beta) * math.sin(math.pi * self.Beta / 2) / (
                math.gamma((1 + self.Beta) / 2) * self.Beta * (2 ** ((self.Beta - 1) / 2)))) ** (1 / self.Beta)


    def produce_step(self, x, gbest):

        for i in range(x.shape[0]):
            
            u = np.random.normal(0, self.sigma_u, 1)
            v = np.random.normal(0, self.sigma_v, 1)
            Ls = u / ((abs(v)) ** (1 / self.Beta))
            stepsize = 0.01 * Ls * (x-gbest)   
            #print(stepsize)
           
        return stepsize
