# encoding=utf8
import random
import numpy
from NiaPy.benchmarks.utility import Utility
import time

__all__ = ['ParticleSwarmAlgorithm']


class ParticleSwarmAlgorithm:
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

    def __init__(self, D, NP, nFES, C1, C2, w, vMin, vMax, benchmark):
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

        #modified bound
        self.Lower_D = self.benchmark.Lower_D  # lower bound
        self.Upper_D = self.benchmark.Upper_D  # upper bound

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

        # record the best solution for each FE
        self.iters = int(self.nFES/self.NP)
        self.gBestFitness_record = numpy.zeros([self.iters*self.NP])

        #parameters compare to IFB-PSO
        self.imp_check = 1000 
        self.iters_check = numpy.zeros([self.NP])
        #improved PSO metrics:
        self.jump_times = 0
        self.jump_improving_times = 0
        self.sum_improving_iters = 0
        self.avg_improving_iters = 0
        self.improved_jump_percentage = 0
        self.jump_mark = numpy.zeros([self.NP])
        self.p_mark_iters = numpy.zeros([self.NP])

    def init(self):
        """Initialize positions."""
        for i in range(self.NP):
            for j in range(self.D):
                self.Solution[i][j] = random.random() * \
                    (self.Upper_D[j] - self.Lower_D[j]) + self.Lower_D[j]

    def eval_true(self):
        """Check evaluations."""

        if self.evaluations == self.nFES:
            self.eval_flag = False

    def bounds(self, position):
        """Keep it within bounds."""
        for i in range(self.D):
            if position[i] < self.Lower_D[i]:
                position[i] = self.Lower_D[i]
            if position[i] > self.Upper_D[i]:
                position[i] = self.Upper_D[i]
        return position

    def move_particles(self):
        """Move particles in search space."""
        self.init()

        iters_count = 0 #count for iterations

        while self.eval_flag is not False:

            #print('PSO iteration:%d' %(iters_count+1))

            for i in range(self.NP):
                self.Solution[i] = self.bounds(self.Solution[i])

                self.eval_true()
                if self.eval_flag is not True:
                    break

                Fit = self.Fun(self.D, self.Solution[i])
                

                if iters_count==0:
                    cur_gBestFitness = Fit
            
                if self.jump_mark[i] == 1:
                    self.p_mark_iters[i] += 1

                if Fit < self.pBestFitness[i]:
                    self.pBestFitness[i] = Fit
                    self.pBestSolution[i] = self.Solution[i]
                    # the fitness value has improved
                    self.iters_check[i] = 0
                    # metrics
                    if self.jump_mark[i] == 1:
                        self.jump_improving_times += 1 #backward particle improved
                        self.sum_improving_iters += (self.p_mark_iters[i])
                        self.p_mark_iters[i] = 0
                        self.jump_mark[i] = 0
                else:
                    #the fitness value has not improved
                    self.iters_check[i] += 1

                if Fit < self.gBestFitness:
                    self.gBestFitness = Fit
                    self.gBestSolution = self.Solution[i]
                    #record to the gBestFitness_record
                    # revise iter to elvaluation 2022/9/29
                    self.gBestFitness_record[self.evaluations] = Fit
                    cur_gBestFitness = Fit
                else:
                    self.gBestFitness_record[self.evaluations] = cur_gBestFitness
                
                self.evaluations = self.evaluations + 1

            for i in range(self.NP):

                if self.iters_check[i] >= self.imp_check: #stuck in local minimum

                    # marked the particle
                    self.jump_mark[i] = 1 #marked the particle
                    self.jump_times += 1 #add jump times
                    self.p_mark_iters[i] = 0 
                    self.iters_check[i] = 0

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
        if self.jump_times != 0 and self.jump_improving_times != 0:     
            self.improved_jump_percentage = self.jump_improving_times / self.jump_times
            self.avg_improving_iters = self.sum_improving_iters / self.jump_improving_times
            print(self.improved_jump_percentage)
            print(self.avg_improving_iters)
        else:
            print('no stuck paticle')

        
        return self.gBestFitness
 
    def run(self):
        """Run."""
        return self.move_particles()