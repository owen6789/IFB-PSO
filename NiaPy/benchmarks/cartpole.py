import math
import numpy as np
import random
import gym
import time

__all__ = ['f_cartpole']

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
class f_cartpole: 

    def __init__(self, Lower=0, Upper=500):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 6
        self.Lower_D = np.array([0,0,0,0,0,0])
        self.Upper_D = np.array([100,100,100,100,100,100])
        self.render_en = False

    def function(self):
        def evaluate(D, sol):

           
            val_t1 = 0.0
            val_t2 = 0.0
            val = 0.0
            env = gym.make('CartPole-v1') 
            done = False
            kp_p = sol[0] 
            kd_p = sol[1] 
            kp_th = sol[2] 
            kd_th = sol[3]
            w1 = (sol[4] / 100  * 20) - 10
            w2 = (sol[5] / 100  * 20) - 10
            dt = 0.02
            
            # +pi/6
            state = env.reset(init_rad = (math.pi/6))
            total_step = 0
            
            while not done:
 
                # position error
                e_p  = state[0]
                v = state[1]
                # theta error
                e_theta = state[2]
                v_theta = state[3]
                u_p =  kp_p * (e_p) + kd_p * (v)
                u_theta = kp_th * (e_theta) + kd_th * (v_theta)
                action = w1 * u_p + w2 * u_theta

                state, reward, done, _ = env.step(action)
                val_t1 += reward
                total_step += 1

            if val_t1 > 40000:
                val_t1 = val_t1 - total_step

            # -pi/6
            state = env.reset(init_rad = (-math.pi/6))
            total_step = 0
            done = False
            
            while not done:
 
                # position error
                e_p  = state[0]
                v = state[1]
                # theta error
                e_theta = state[2]
                v_theta = state[3]
                u_p =  kp_p * (e_p) + kd_p * (v)
                u_theta = kp_th * (e_theta) + kd_th * (v_theta)
                action = w1 * u_p + w2 * u_theta

                state, reward, done, _ = env.step(action)
                val_t2 += reward
                total_step += 1

            if val_t2 > 40000:
                val_t2 = val_t2 - total_step

            env.close()
            
            val = val_t1 + val_t2
            
            return val

        return evaluate   

