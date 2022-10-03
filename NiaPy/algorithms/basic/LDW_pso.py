# encoding=utf8
# Linear Decreasing Inertia Weight
import random
import numpy
from NiaPy.benchmarks.utility import Utility

__all__ = ['LDWParticleSwarmAlgorithm']


class LDWParticleSwarmAlgorithm:
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

        #record the best solution for each FE
        self.iters = int(self.nFES/self.NP)
        self.gBestFitness_record = numpy.zeros([self.iters*self.NP])

        self.iter_max = self.iters
        self.w_max = 0.9
        self.w_min = 0.4

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

        while self.eval_flag is not False:

            #print('PSO iteration:%d' %(iters_count+1))

            for i in range(self.NP):
                self.Solution[i] = self.bounds(self.Solution[i])

                self.eval_true()
                if self.eval_flag is not True:
                    break

                Fit = self.Fun(self.D, self.Solution[i])

                if Fit < self.pBestFitness[i]:
                    self.pBestFitness[i] = Fit
                    self.pBestSolution[i] = self.Solution[i]

                if Fit < self.gBestFitness:
                    self.gBestFitness = Fit
                    self.gBestSolution = self.Solution[i]
                    #record to the gBestFitness_record
                    self.gBestFitness_record[self.evaluations] = Fit
                    cur_gBestFitness = Fit
                else:
                    self.gBestFitness_record[self.evaluations] = cur_gBestFitness
                
                self.evaluations = self.evaluations + 1

            self.w = ((self.iter_max-iters_count+1)*(self.w_max-self.w_min))/(self.iter_max) + self.w_min

            for i in range(self.NP):
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