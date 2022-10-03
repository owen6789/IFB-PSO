# encoding=utf8
import random
import numpy
from NiaPy.benchmarks.utility import Utility
from sklearn.ensemble import IsolationForest
import time
import sys
from NiaPy.algorithms.basic import Levy_Flight

__all__ = ['LevyFlightParticleSwarmAlgorithm']


class LevyFlightParticleSwarmAlgorithm:
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

    def __init__(self, D, NP, nFES, C1, C2, w, vMin, vMax, benchmark, imp_check, strategy):
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
        
        #record the best solution for each iters
        self.iters = int(self.nFES/self.NP)
        self.gBestFitness_record = numpy.zeros([self.iters])
        self.time_record = numpy.zeros([self.iters])
        self.time_sum = 0
        self.gBest_p = numpy.zeros([self.iters,self.D])

        #======================= modified part ====================================
        # use Isolation Tree to seperate the anomaly
        # check the particles whether stucking local minimum
        # if the fitness value cannot improve more than imp_check iterations
        # => view as stucking in local minimum
        self.strategy = strategy
        self.imp_check = imp_check 
        self.iters_check = numpy.zeros([self.NP])
        # record the history of particles
        # [particle no.][iterations][particle]
        self.p_his = numpy.zeros([self.NP, self.iters, self.D]) 
        # improved PSO metrics:
        self.jump_times = 0
        self.jump_improving_times = 0
        self.sum_improving_iters = 0
        self.avg_improving_iters = 0
        self.improved_jump_percentage = 0
        self.jump_mark = numpy.zeros([self.NP])
        self.p_mark_iters = numpy.zeros([self.NP])
        # parameters for Levy Flight PSO
        # array for the particles' state
        self.levy_motion = Levy_Flight.Levy_Flight()
        self.max_iter = 10000

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

        iters_count = 0 # count for iterations

        while self.eval_flag is not False:

            iter_time_eval = 0
            self.w = (self.max_iter - iters_count) / self.max_iter
            
            for i in range(self.NP): # i: no. of particle
                self.Solution[i] = self.bounds(self.Solution[i])

                self.eval_true()
                if self.eval_flag is not True:
                    break
                
                time_eval_start = time.time()
                Fit = self.Fun(self.D, self.Solution[i])
                time_eval_end = time.time()
                iter_time_eval += (time_eval_end-time_eval_start)
                self.evaluations = self.evaluations + 1

                if self.jump_mark[i] == 1:
                    self.p_mark_iters[i] += 1

                # record the particle to the historical array
                self.p_his[i][iters_count] = self.Solution[i]

                if Fit < self.pBestFitness[i]:
                    self.pBestFitness[i] = Fit
                    self.pBestSolution[i] = self.Solution[i]
                    # the fitness value has improved
                    self.iters_check[i] = 0
                    # metrics
                    if self.jump_mark[i] == 1:
                        self.jump_improving_times += 1 # backward particle improved
                        self.sum_improving_iters += (self.p_mark_iters[i])
                        self.p_mark_iters[i] = 0
                        self.jump_mark[i] = 0
                else:
                    # the fitness value has not improved
                    self.iters_check[i] += 1

                if Fit < self.gBestFitness:
                    self.gBestFitness = Fit
                    self.gBestSolution = self.Solution[i]
                    # record to the gBestFitness_record
                    # record the best position
                    self.gBest_p[iters_count] = self.gBestSolution
                    self.gBestFitness_record[iters_count] = Fit
                    cur_gBestFitness = Fit
                else:
                    self.gBestFitness_record[iters_count] = cur_gBestFitness

            # update particle's position
            for i in range(self.NP):
                # the particle cannot improve for more than imp_check iterations
                if self.iters_check[i] >= self.imp_check:
                    # do Levy Flight
                    # update Beta and sigma_u
                    self.levy_motion.update_sigma_u()
                    Levy_step = self.levy_motion.produce_step(self.Solution[i],self.gBestSolution)
                    self.Solution[i] = self.Solution[i] + Levy_step

                else:
                    
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

            iters_count += 1 

        return self.gBestFitness

    def run(self):
        """Run."""
        return self.move_particles()
