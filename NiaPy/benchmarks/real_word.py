#real world optimization problems
import math
import numpy as np
import random

__all__ = ['p1']

class p1:

    def __init__(self, Lower=-6.4, Upper=6.35):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([6]) * self.Lower
        self.Upper_D = np.ones([6]) * self.Upper
    

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            
            theta = 2*math.pi/100
            f = 0

            for t in range(101):

                y_t = sol[0] * math.sin(sol[1]*t*theta + sol[2]*math.sin(sol[3]*t*theta + sol[4]*math.sin(sol[5]*t*theta)))
                y_0_t=1*math.sin(5*t*theta-1.5*math.sin(4.8*t*theta+2*math.sin(4.9*t*theta)))
                f = f + (y_t-y_0_t)**2

            return f

        return evaluate  
