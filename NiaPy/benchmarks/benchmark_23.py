# 23 benchmark function in paper: Evolutionary Programming Made Faster
import math
import numpy as np
import random

__all__ = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',\
           'f11','f12','f13','f14','f15','f16'\
           ,'f17','f18','f19','f20','f21','f22','f23']

#Sphere Model
class f1: 

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.pow(sol[i], 2)

            return val

        return evaluate   

#Schwefel's Problem 2.22
class f2:

    def __init__(self, Lower=-10.0, Upper=10.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper
    
    @classmethod
    def function(cls):
        def evaluate(D, sol):

            part1 = 0.0
            part2 = 1.0

            for i in range(D):
                part1 += abs(sol[i])
                part2 *= abs(sol[i])

            return part1 + part2

        return evaluate

#Schwefel's Problem 1.2
class f3:

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper
    
    @classmethod
    def function(cls):
        def evaluate(D, sol):

            x = np.array(sol)
            
            return np.sum([np.sum(x[:i]) ** 2 for i in range(len(x))])

        return evaluate

#Schwefel's Problem 2.21
class f4:

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            maximum = 0.0

            for i in range(D):
                if abs(sol[i]) > maximum:
                    maximum = abs(sol[i])

            return maximum

        return evaluate

#Generalized Rosenbrock's Function
class f5:

    def __init__(self, Lower=-30.0, Upper=30.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D - 1):
                val += 100.0 * math.pow(sol[i + 1] - math.pow((sol[i]), 2),
                                        2) + math.pow((sol[i] - 1), 2)

            return val

        return evaluate
#Step function
class f6:

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.pow(math.floor(sol[i] + 0.5), 2)

            return val

        return evaluate

#Quartic Function i.e. Noise
class f7:

    def __init__(self, Lower=-1.28, Upper=1.28):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val= 0.0

            for i in range(D):
                
                val += (i*math.pow(sol[i],4))

            return val+ random.random()
        
        return evaluate

#Generalized Schwefel's Problem 2.26
class f8:

    def __init__(self, Lower=-500.0, Upper=500.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += (sol[i] * math.sin(math.sqrt(abs(sol[i]))))

            return (-val)

        return evaluate

#Generalized Rastrigin's Function
class f9:

    def __init__(self, Lower=-5.12, Upper=5.12):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.pow(sol[i], 2) - (10.0 * math.cos(2 * math.pi * sol[i]))

            return 10 * D + val

        return evaluate

#Ackley's Function
class f10:

    def __init__(self, Lower=-32.768, Upper=32.768):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        """Return benchmark evaluation function."""
        def evaluate(D, sol):

            a = 20  # Recommended variable value
            b = 0.2  # Recommended variable value
            c = 2 * math.pi  # Recommended variable value

            val = 0.0
            val1 = 0.0
            val2 = 0.0

            for i in range(D):
                val1 += math.pow(sol[i], 2)
                val2 += math.cos(c * sol[i])

            temp1 = -b * math.sqrt(val1 / D)
            temp2 = val2 / D

            val = -a * math.exp(temp1) - math.exp(temp2) + a + math.exp(1)

            return val

        return evaluate
    
#Generalized Griewank Function
class f11:

    def __init__(self, Lower=-600.0, Upper=600.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val1 = 0.0
            val2 = 1.0

            for i in range(D):
                val1 += (math.pow(sol[i], 2) / 4000.0)
                val2 *= (math.cos(sol[i] / math.sqrt(i + 1)))

            return val1 - val2 + 1.0

        return evaluate


#Generalized Penalized Function
class f12:
    def __init__(self, Lower=-50.0, Upper=50.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            
            def y(i): #i:index
                
                yi = 1 + 0.25*(sol[i]+1) 

                return yi
            
            def u(i,a,k,m):

                if sol[i] > a:
                    ui = k * math.pow((sol[i]-a),m)
                elif sol[i] < -a:
                    ui = k * math.pow((-sol[i]-a),m)
                else:
                    ui = 0

                return ui
            
            val = 0
            val1 = 0
            val2 = 0

            for j in range(D-1):

                val += (math.pow((y(j)-1),2) * (1+10*math.pow((math.sin(math.pi*y(j+1))),2)))

            val1 = (math.pi/D) * (10*math.pow((math.pi*y(0)),2) + val + math.pow((y(D-1)-1),2))

            for j in range(D):
                
                val2 += u(j,5,100,4)

            return val1 + val2

        return evaluate

class f13:
    def __init__(self, Lower=-50.0, Upper=50.0):
        self.Lower = Lower
        self.Upper = Upper
        self.D = 100
        self.Lower_D = np.ones([self.D]) * self.Lower
        self.Upper_D = np.ones([self.D]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            def u(i,a,k,m):

                if sol[i] > a:
                    ui = k * math.pow((sol[i]-a),m)
                elif sol[i] < -a:
                    ui = k * math.pow((-sol[i]-a),m)
                else:
                    ui = 0

                return ui
            
            val = 0
            val1 = 0
            val2 = 0

            for j in range(D-1):

                val += (math.pow(sol[j]-1,2)*(1+math.pow(math.sin(3*math.pi*sol[j+1]),2)))
            
            val1 = 0.1*(math.pow(math.sin(math.pi*3*sol[0]),2) + \
                    val + math.pow(sol[D-1]-1,2))*(1+math.pow(2*math.pi*sol[D-1],2))
            
            for j in range(D):

                val2 += u(j,5,100,4)

            return val1 + val2

        return evaluate    

#Shekel's Foxholes Function
class f14:

    def __init__(self, Lower=-65.536, Upper=65.536):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([2]) * self.Lower
        self.Upper_D = np.ones([2]) * self.Upper
    
    @classmethod
    def function(cls):
        def evaluate(D,sol):
            
            a = np.array([[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],\
                          [-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]])
            
            val = 0

            for j in range(1,26):

                val += (1/(j+math.pow(sol[0]-a[0][j-1],6)+math.pow(sol[1]-a[1][j-1],6))) 

            result = math.pow(((1/500)+val),-1)

            return result

        return evaluate            
    

#Kowalik's Function
class f15:
    def __init__(self, Lower=-5.0, Upper=5.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([4]) * self.Lower
        self.Upper_D = np.ones([4]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D,sol):

            a = np.array([0.1957,0.1947,0.1735,0.1600,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
            b = np.array([4,2,1,(1/2),(1/4),(1/6),(1/8),(1/10),(1/12),(1/14),(1/16)])

            val = 0

            for i in range(11):
                val += math.pow((a[i]-((sol[0]*(b[i]*b[i]+b[i]*sol[1]))/(b[i]*b[i]+b[i]*sol[2]+sol[3]))),2)

            return val

        return evaluate

#Six-Hump Camel-Back Function
class f16:
    def __init__(self, Lower=-5.0, Upper=5.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([2]) * self.Lower
        self.Upper_D = np.ones([2]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D,sol):

            result = 4*sol[0]*sol[0] - 2.1*math.pow(sol[0],4) + (1/3)*math.pow(sol[0],6) + \
                sol[0]*sol[1] - 4*sol[1]*sol[1] + 4*math.pow(sol[1],4)  

            return result

        return evaluate

#Branin Function
class f17:
    def __init__(self, Lower=-5.0, Upper=15.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.array([-5,0])
        self.Upper_D = np.array([10,15])
    
    @classmethod
    def function(cls):
        def evaluate(D,sol):

            x1 = sol[0]
            x2 = sol[1]

            result = math.pow(x2-(5.1/4/math.pow(math.pi,2)*x1*x1)+5*x1/math.pi-6, 2) + \
                    10*(1-1/(8*math.pi))*math.cos(x1)+10

            return result

        return evaluate

#Goldstein-Price Function
class f18:
    def __init__(self, Lower=-2.0, Upper=2.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([2]) * self.Lower
        self.Upper_D = np.ones([2]) * self.Upper

    @classmethod
    def function(cls):
        def evaluate(D,sol):

            x1 = sol[0]
            x2 = sol[1]

            result = (1+math.pow(x1+x2+1,2)*(19-14*x1+3*x1*x1-14*x2+6*x1*x2+3*x2*x2)) * \
                (30+math.pow(2*x1-3*x2,2)*(18-32*x1+12*x1*x1+48*x2-36*x1*x2+27*x2*x2))

            return result
        
        return evaluate

#Hartman's Family
#f19: n=3 f20: n=6
class f19:
    def __init__(self, Lower=-2.0, Upper=2.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([3]) * self.Lower
        self.Upper_D = np.ones([3]) * self.Upper

    @classmethod
    def function(cls):

        def evaluate(D,sol):

            a = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
            c = np.array([1,1.2,3,3.2])
            p = np.array([[0.3689,0.1170,0.2673],[0.4699,0.4387,0.7470],\
                        [0.1091,0.8732,0.5547],[0.038150,0.5743,0.8828]])

            result = 0

            for i in range(4):

                val = 0

                for j in range(3):

                    val += (a[i][j]*(sol[j]-p[i][j])**2)

                result += (c[i]*math.exp(-val))

            return -result
        
        return evaluate

class f20:
    def __init__(self, Lower=-2.0, Upper=2.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([6]) * self.Lower
        self.Upper_D = np.ones([6]) * self.Upper

    @classmethod
    def function(cls):

        def evaluate(D,sol):

            a = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
            c = np.array([1,1.2,3,3.2])
            p = np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],[0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],\
                         [0.2348,0.1415,0.3522,0.2883,0.3047,0.6650],[0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
            result = 0
            
            for i in range(4):

                val = 0

                for j in range(3):

                    val += (a[i][j]*(sol[j]-p[i][j])**2)

                result += (c[i]*math.exp(-val))

            return -result
        
        return evaluate

#Shekel's Family
class f21: 
    def __init__(self, Lower=0.0, Upper=10.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([4]) * self.Lower
        self.Upper_D = np.ones([4]) * self.Upper
 

    @classmethod
    def function(cls):

        def evaluate(D,sol):

            a = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],\
                          [3,7,3,7],[2,9,2,9],[5,5,3,3],\
                         [8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])

            c = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
            
            m = 5
            val = 0
            for i in range(m):

                val += math.pow(np.inner(sol-a[i],np.transpose(sol-a[i])) + c[i], -1)

            return -val

        return evaluate

class f22: 
    def __init__(self, Lower=0.0, Upper=10.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([4]) * self.Lower
        self.Upper_D = np.ones([4]) * self.Upper
 

    @classmethod
    def function(cls):

        def evaluate(D,sol):

            a = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],\
                          [3,7,3,7],[2,9,2,9],[5,5,3,3],\
                         [8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])

            c = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
            
            m = 7
            val = 0
            for i in range(m):

                val += math.pow(np.inner(sol-a[i],np.transpose(sol-a[i])) + c[i], -1)

            return -val

        return evaluate

class f23: 
    def __init__(self, Lower=0.0, Upper=10.0):
        self.Lower = Lower
        self.Upper = Upper
        self.Lower_D = np.ones([4]) * self.Lower
        self.Upper_D = np.ones([4]) * self.Upper
 

    @classmethod
    def function(cls):

        def evaluate(D,sol):

            a = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],\
                          [3,7,3,7],[2,9,2,9],[5,5,3,3],\
                         [8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])

            c = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
            
            m = 10
            val = 0

            for i in range(m):

                val += math.pow(np.inner(sol-a[i],np.transpose(sol-a[i])) + c[i], -1)

            return -val

        return evaluate