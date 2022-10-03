# encoding=utf8
import random
import numpy
from NiaPy.benchmarks.utility import Utility
from sklearn.ensemble import IsolationForest

__all__ = ['BRWParticleSwarmAlgorithm']


class BRWParticleSwarmAlgorithm:
    r"""Implementation of Particle Swarm Optimization algorithm.

    **Algorithm:** Particle Swarm Optimization algorithm

    **Date:** 2018

    **Authors:** Lucija Brezočnik, Grega Vrbančič, and Iztok Fister Jr.

    **License:** MIT

    **Reference paper:**
        Kennedy, J. and Eberhart, R. "Particle Swarm Optimization".
        Proceedings of IEEE International Conference on Neural Networks.
        IV. pp. 1942--1948, 1995.
    """

    def __init__(self, D, NP, nFES, C1, C2, w, vMin, vMax, benchmark,imp_check,strategy):
        r"""**__init__(self, NP, D, nFES, C1, C2, w, vMin, vMax, benchmark)**.

        Arguments:
            NP {integer} -- population size

            D {integer} -- dimension of problem

            nFES {integer} -- number of function evaluations

            C1 {decimal} -- cognitive component

            C2 {decimal} -- social component

            w {decimal} -- inertia weight

            vMin {decimal} -- minimal velocity

            vMax {decimal} -- maximal velocity

            benchmark {object} -- benchmark implementation object

            imp_check{integer} -- the threshold of iteration of the particle which cannot improve itself

            strategy{integer} -- 1: isolation forest backward 2: random backward 3: jump to random position

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.NP = NP  # population size; number of search agents
        self.D = D  # dimension of the problem
        self.C1 = C1  # cognitive component
        self.C2 = C2  # social component
        self.w = w  # inertia weight
        self.vMin = vMin  # minimal velocity
        self.vMax = vMax  # maximal velocity
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.nFES = nFES  # number of function evaluations
        self.eval_flag = True  # evaluations flag
        self.evaluations = 0  # evaluations counter
        self.Fun = self.benchmark.function()

        self.Solution = numpy.zeros((self.NP, self.D))  # positions of search agents
        self.Velocity = numpy.zeros((self.NP, self.D))  # velocities of search agents

        self.pBestFitness = numpy.zeros(self.NP)  # personal best fitness
        self.pBestFitness.fill(float("inf"))
        self.pBestSolution = numpy.zeros((self.NP, self.D))  # personal best solution

        self.gBestFitness = float("inf")  # global best fitness
        self.gBestSolution = numpy.zeros(self.D)  # global best solution
        
        #record the best solution for each iters
        self.iters = int(self.nFES/self.NP)
        self.gBestFitness_record = numpy.zeros([self.iters])

        #=======================modified part====================================
        #use isolation tree to seperate the anomaly

        #check the particles whether stucking local minimum
        #if the fitness value cannot improve more than imp_check iterations
        #=> view as stucking in local minimum
        self.strategy = strategy
        self.imp_check = imp_check 
        self.iters_check = numpy.zeros([self.NP])
        #record the history of particles
        #[particle no.][iterations][particle]
        self.p_his = numpy.zeros([self.NP, self.iters, self.D]) 

    def init(self):
        """Initialize positions."""
        for i in range(self.NP):
            for j in range(self.D):
                self.Solution[i][j] = random.random() * \
                    (self.Upper - self.Lower) + self.Lower

    def eval_true(self):
        """Check evaluations."""

        if self.evaluations == self.nFES:
            self.eval_flag = False

    def bounds(self, position):
        """Keep it within bounds."""
        for i in range(self.D):
            if position[i] < self.Lower:
                position[i] = self.Lower
            if position[i] > self.Upper:
                position[i] = self.Upper
        return position

    def move_particles(self):
        """Move particles in search space."""
        self.init()

        iters_count = 0 #count for iterations
        #actually count = iters_count + 1

        while self.eval_flag is not False:

            #print('IFB-PSO iteration:%d' %(iters_count+1))

            for i in range(self.NP): #i: no. of particle
                self.Solution[i] = self.bounds(self.Solution[i])

                self.eval_true()
                if self.eval_flag is not True:
                    break

                Fit = self.Fun(self.D, self.Solution[i])
                self.evaluations = self.evaluations + 1

                #record the particle to the historical array
                self.p_his[i][iters_count] = self.Solution[i]

                if Fit < self.pBestFitness[i]:
                    self.pBestFitness[i] = Fit
                    self.pBestSolution[i] = self.Solution[i]
                    #the fitness value has improved
                    self.iters_check[i] = 0
                else:
                    #the fitness value has not improved
                    self.iters_check[i] += 1

                if Fit < self.gBestFitness:
                    self.gBestFitness = Fit
                    self.gBestSolution = self.Solution[i]
                    #record to the gBestFitness_record
                    self.gBestFitness_record[iters_count] = Fit
                    cur_gBestFitness = Fit
                else:
                    self.gBestFitness_record[iters_count] = cur_gBestFitness

            #update particle's position
            for i in range(self.NP):
                #the particle cannot improve for more than imp_check iterations
                if self.iters_check[i] >= self.imp_check:

                    #print('#######start backward process######')
                    #try to jump out local minimum

                    #using iForest to choose the anamoly
                    if self.strategy == 1:
                        #get the historical particles
                        #i: no of p , 0:iters_count+1:history of p until current iters
                        X = self.p_his[i][0:iters_count+1]
                        #print(X.shape)
                        #isolation forest
                        clf = IsolationForest(random_state=0,contamination='auto').fit(X)
                        anamoly_score = (clf.score_samples(X))*(-1) #get s
                        #print(anamoly_score)
                        #use anamoly score to determine back to which point
                        max_score = 0
                        p_no = -1

                        for j in range(iters_count+1):
                        
                            if anamoly_score[j] > max_score:
                                p_no = j
                                max_score = anamoly_score[j]
                        
                        self.Solution[i] = X[p_no]

                        #add random variation 0.9~1.1
                        # for j in range(self.D):
                        #     self.Solution[i][j] = self.Solution[i][j]*(1+(random.random()-0.5)/5)
                        
                        #check bound
                        self.Solution[i] = self.bounds(self.Solution[i])

                    #random backward 
                    elif self.strategy == 2:
                        
                        X = self.p_his[i][0:iters_count+1]
                        p_back_no = int(random.random()*iters_count)
                        self.Solution[i] = X[p_back_no]

                    #random position
                    elif self.strategy == 3:
                        
                        for j in range(self.D):
                            self.Solution[i][j] = random.random() * \
                                                 (self.Upper - self.Lower) + self.Lower
                    
                    #reset the count to 0
                    self.iters_check[i] = 0


                else:
                    
                    self.w = 0.5 + 0.5*(random.random())

                    for j in range(self.D):

                        self.Velocity[i][j] = (self.w * self.Velocity[i][j])+ \
                            (self.C1 * random.random() * (self.pBestSolution[i][j] - self.Solution[i][j])) + \
                            (self.C2 * random.random() * (self.gBestSolution[j] - self.Solution[i][j]))

                        if self.Velocity[i][j] < self.vMin:
                            self.Velocity[i][j] = self.vMin
                        if self.Velocity[i][j] > self.vMax:
                            self.Velocity[i][j] = self.vMax

                        self.Solution[i][j] = self.Solution[i][j] + \
                            self.Velocity[i][j]

            #iterations add 1 
            iters_count += 1 

        return self.gBestFitness

    def run(self):
        """Run."""
        return self.move_particles()
