import math
import numpy as np
import random
import gym
import time

__all__ = ['f_mountaincar']

class PID():
    def __init__(self,KP,KI,KD):
        self.p = KP
        self.i = KI
        self.d = KD
        self.integral = 0
        self.prev = 0
    
    def action(self,error,dt):
        result = self.p * error + self.i * self.integral + (self.d * (error-self.prev))/dt
        self.prev = error
        self.integral = self.integral + error
        return result

#Sphere Model
class f_mountaincar: 

    def __init__(self, Lower=0, Upper=15):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 5
        self.Lower_D = np.array([0,0,0,0,0])
        self.Upper_D = np.array([15,15,15,15,15])

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0
            iters = 0
            err_sum = 0
            env = gym.make('MountainCarContinuous-v0') 
            state = env.reset()
            done = False
            pid = PID(sol[0],sol[1],sol[2])
            dt = 0.02
            cmd = 0
            w1 = (sol[3]/15)*20 - 10
            w2 = (sol[4]/15)*20 - 10
            #th = (sol[3]/15)*10 #0~10

            while not done:
                tic = time.perf_counter()
                error = w1*state[0] + w2*state[1] 
                err_sum += (abs(state[0])+abs(state[1]))
                action =  pid.action(error,dt)
                cmd = np.array([action])
                
                #env.render()
                state, reward, done, _ = env.step(cmd)
                iters += 1
                #val += abs(state[0] - (-0.5))

            env.close()
            val = (-reward)
            return val
            #print(val)
            # if val == 100000.0:
            #     avg_err = err_sum/100000
            #     val += (1/avg_err)

            # if iters == 200:
            #     #print(456)
            #     return -val
            # else:
            #     val += 200 * (200/iters)
            #     #print(123)
            #     return -val

        return evaluate   

