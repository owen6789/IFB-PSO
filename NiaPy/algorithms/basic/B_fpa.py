import random
import numpy as np
from scipy.special import gamma as Gamma
from NiaPy.benchmarks.utility import Utility
from sklearn.ensemble import IsolationForest

__all__ = ['BackwardFlowerPollinationAlgorithm']


class BackwardFlowerPollinationAlgorithm:
    r"""Implementation of Flower Pollination algorithm.

    **Algorithm:** Flower Pollination algorithm

    **Date:** 2018

    **Authors:** Dusan Fister & Iztok Fister Jr.

    **License:** MIT

    **Reference paper:**
        Yang, Xin-She. "Flower pollination algorithm for global optimization."
        International conference on unconventional computing and natural computation.
        Springer, Berlin, Heidelberg, 2012.

    Implementation is based on the following MATLAB code:
    https://www.mathworks.com/matlabcentral/fileexchange/45112-flower-pollination-algorithm?requestedDomain=true
    """

    def __init__(self, D, NP, nFES, p, benchmark, imp_check):
        r"""**__init__(self, D, NP, nFES, p, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            p {decimal} -- probability switch

            benchmark {object} -- benchmark implementation object

        Raises:
            TypeError -- Raised when given benchmark function which does not exists.

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.D = D  # dimension
        self.NP = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.p = p  # probability switch
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.Fun = self.benchmark.function()  # function
        #modified bound
        self.Lower_D = self.benchmark.Lower_D  # lower bound
        self.Upper_D = self.benchmark.Upper_D  # upper bound

        self.f_min = 0.0  # minimum fitness

        self.Lb = self.Lower_D # lower bound
        self.Ub = self.Upper_D  # upper bound

        self.dS = [[0 for _i in range(self.D)]
                   for _j in range(self.NP)]  # differential
        self.Sol = [[0 for _i in range(self.D)]
                    for _j in range(self.NP)]  # population of solutions
        self.Fitness = [0] * self.NP  # fitness
        self.best = [0] * self.D  # best solution
        self.eval_flag = True  # evaluations flag
        self.evaluations = 0  # evaluations counter

        self.iters = int(self.nFES/self.NP)
        self.gBestFitness_record = np.zeros([self.iters])
        self.iters_count = 0
        #=======================modified part====================================
        #use isolation tree to seperate the anomaly
        #check the particles whether stucking local minimum
        #if the fitness value cannot improve more than imp_check iterations
        #=> view as stucking in local minimum
        self.imp_check = imp_check 
        self.iters_check = np.zeros([self.NP])
        #record the history of particles
        #p_his:[particle no.][iterations]
        self.p_his = np.zeros([self.NP, self.iters, self.D]) 
        self.p_best = np.zeros([self.NP])


    def best_flower(self):
        """Check best solution."""
        i = 0
        j = 0
        for i in range(self.NP):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.D):
            self.best[i] = self.Sol[j][i]
        self.f_min = self.Fitness[j]

    def eval_true(self):
        """Check evaluations."""

        if self.evaluations == self.nFES:
            self.eval_flag = False

    @classmethod
    def simplebounds(cls, val, lower, upper):
        """Keep it within bounds."""
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def init_flower(self):
        """Initialize flowers."""
        for i in range(self.D):
            self.Lb[i] = self.Lower
            self.Ub[i] = self.Upper

        for i in range(self.NP):
            for j in range(self.D):
                rnd = random.uniform(0, 1)
                self.dS[i][j] = 0.0
                self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]) * rnd
            self.Fitness[i] = self.Fun(self.D, self.Sol[i])
            self.evaluations = self.evaluations + 1
        self.best_flower()

    def move_flower(self):
        """Move in search space."""
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)]
        self.init_flower()

        while self.eval_flag is not False:
            for i in range(self.NP):
                if random.uniform(0, 1) > self.p:  # probability switch
                    L = self.Levy()
                    for j in range(self.D):
                        self.dS[i][j] = L[j] * (self.Sol[i][j] - self.best[j])
                        S[i][j] = self.Sol[i][j] + self.dS[i][j]

                        S[i][j] = self.simplebounds(
                            S[i][j], self.Lb[j], self.Ub[j])
                else:
                    epsilon = random.uniform(0, 1)
                    JK = np.random.permutation(self.NP)

                    for j in range(self.D):
                        S[i][j] = S[i][j] + epsilon * \
                            (self.Sol[JK[0]][j] - self.Sol[JK[1]][j])
                        S[i][j] = self.simplebounds(
                            S[i][j], self.Lb[j], self.Ub[j])

                self.eval_true()
                if self.eval_flag is not True:
                    break

                Fnew = self.Fun(self.D, S[i])
                self.evaluations = self.evaluations + 1

                #record the candidate of solution
                self.p_his[i][self.iters_count] = S[i]
                #check the fitness whether improved
                if Fnew < self.Fitness[i]:
                    self.iters_check[i] = 0
                else:
                    self.iters_check[i] += 1

                if Fnew <= self.Fitness[i]:
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew

                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = S[i][j]
                    self.f_min = Fnew

                #start backward process
                if self.iters_check[i] >= self.imp_check:

                    #print('#######start backward process######')
                    #print(self.iters_check[i])
                    #try to jump out local minimum
                    X = self.p_his[i][0:self.iters_count+1]
                    clf = IsolationForest(random_state=0,contamination='auto').fit(X)
                    anamoly_score = (clf.score_samples(X))*(-1) #get s
                    #print(anamoly_score)
                    #use anamoly score to determine back to which point
                    max_score = 0
                    p_no = -1
                    
                    for j in range(self.iters_count+1):
                    
                        if anamoly_score[j] > max_score:
                            p_no = j
                            max_score = anamoly_score[j]
                    
                    self.Sol[i] = X[p_no]
                    self.iters_check[i] = 0

            self.gBestFitness_record[self.iters_count]= self.f_min
            #print(self.iters_count)
            #print(self.iters_check)
            #print(self.f_min)
            self.iters_count+=1

        return self.f_min

    def Levy(self):
        """Levy flight."""
        beta = 1.5
        sigma = (Gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (Gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = [[0] for j in range(self.D)]
        v = [[0] for j in range(self.D)]
        step = [[0] for j in range(self.D)]
        L = [[0] for j in range(self.D)]

        for j in range(self.D):
            u[j] = np.random.normal(0, 1) * sigma
            v[j] = np.random.normal(0, 1)
            step[j] = u[j] / abs(v[j])**(1 / beta)
            L[j] = 0.01 * step[j]

        return L

    def run(self):
        """Run."""
        return self.move_flower()
