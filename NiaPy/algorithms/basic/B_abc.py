import random as rnd
import copy
import numpy as np
from NiaPy.benchmarks.utility import Utility

__all__ = ['ArtificialBeeColonyAlgorithm']


class SolutionABC:

    def __init__(self, D, LB, UB):
        self.D = D
        self.Solution = []
        self.Fitness = float('inf')
        self.LB = LB
        self.UB = UB
        self.generateSolution()

    def generateSolution(self):
        self.Solution = [self.LB + (self.UB - self.LB) * rnd.random()
                         for _i in range(self.D)]

    def repair(self):
        for i in range(self.D):
            if self.Solution[i] > self.UB:
                self.Solution[i] = self.UB

            if self.Solution[i] < self.LB:
                self.Solution[i] = self.LB

    def evaluate(self):
        self.Fitness = SolutionABC.FuncEval(self.D, self.Solution)

    def toString(self):
        pass


class ArtificialBeeColonyAlgorithm:
    r"""Implementation of Artificial Bee Colony algorithm.

    **Algorithm:** Artificial Bee Colony algorithm

    **Date:** 2018

    **Author:** Uros Mlakar

    **License:** MIT

    **Reference paper:**
        Karaboga, D., and Bahriye B. "A powerful and efficient algorithm for
        numerical function optimization: artificial bee colony (ABC) algorithm."
        Journal of global optimization 39.3 (2007): 459-471.

    """

    def __init__(self, D, NP, nFES, benchmark):
        """**__init__(self, D, NP, nFES, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            benchmark {object} -- benchmark implementation object

        Raises:
            TypeError -- Raised when given benchmark function which does not exists.

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.D = D  # dimension of the problem
        self.NP = NP  # population size; number of search agents
        self.FoodNumber = int(self.NP / 2)
        self.Limit = 100
        self.Trial = []  # trials
        self.Foods = []  # foods
        self.Probs = []  # probs
        self.nFES = nFES  # number of function evaluations
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound

        self.FEs = 0
        self.Done = False

        SolutionABC.FuncEval = staticmethod(self.benchmark.function())
        self.Best = SolutionABC(self.D, self.Lower, self.Upper)

        #record the best solution for each iter
        self.iters_count = 0
        self.iters = int(self.nFES/self.NP)
        self.gBestFitness_record = np.zeros([self.iters])

        #Backward ABC part:
        self.food_imp_count = np.zeros(self.FoodNumber)
        self.food_his = np.zeros([self.FoodNumber, self.iters, self.D])

    def init(self):
        """Initialize positions."""
        self.Probs = [0 for i in range(self.FoodNumber)]
        self.Trial = [0 for i in range(self.FoodNumber)]
        for i in range(self.FoodNumber):
            self.Foods.append(SolutionABC(self.D, self.Lower, self.Upper))
            self.Foods[i].evaluate()
            self.FEs += 1
            self.checkForBest(self.Foods[i])

    def CalculateProbs(self):
        """Calculate probs."""
        self.Probs = [1.0 / (self.Foods[i].Fitness + 0.01)
                      for i in range(self.FoodNumber)]
        s = sum(self.Probs)
        self.Probs = [self.Probs[i] / s for i in range(self.FoodNumber)]

    def checkForBest(self, Solution):
        """Check best solution."""
        if Solution.Fitness <= self.Best.Fitness:
            self.Best = copy.deepcopy(Solution)

    def tryEval(self, b):
        """Check evaluations."""
        if self.FEs <= self.nFES:
            b.evaluate()
            self.FEs += 1
        else:
            self.Done = True

    def tryEval2(self, b):
        """Check evaluations."""
        if self.FEs <= self.nFES:
            b.evaluate()
            
        else:
            self.Done = True

    def run(self):
        """Run."""
        self.init()
        self.FEs = self.FoodNumber

        while not self.Done:
            self.Best.toString()
            for i in range(self.FoodNumber):

                #record the food position
                for j in range(self.FoodNumber):
                    #record the solution of food
                    self.food_his[i][self.iters_count] = self.Foods[i].Solution

                newSolution = copy.deepcopy(self.Foods[i])
                param2change = int(rnd.random() * self.D)
                neighbor = int(self.FoodNumber * rnd.random())
                newSolution.Solution[param2change] = self.Foods[i].Solution[param2change] \
                    + (-1 + 2 * rnd.random()) * \
                    (self.Foods[i].Solution[param2change] -
                     self.Foods[neighbor].Solution[param2change])
                newSolution.repair()
                self.tryEval(newSolution)
                if newSolution.Fitness < self.Foods[i].Fitness:
                    self.checkForBest(newSolution)
                    self.Foods[i] = newSolution
                    self.Trial[i] = 0
                else:
                    self.Trial[i] += 1

            self.CalculateProbs()
            t, s = 0, 0
            while t < self.FoodNumber:
                if rnd.random() < self.Probs[s]:
                    t += 1
                    Solution = copy.deepcopy(self.Foods[s])
                    param2change = int(rnd.random() * self.D)
                    neighbor = int(self.FoodNumber * rnd.random())
                    while neighbor == s:
                        neighbor = int(self.FoodNumber * rnd.random())
                    Solution.Solution[param2change] = self.Foods[s].Solution[param2change] \
                        + (-1 + 2 * rnd.random()) * (
                            self.Foods[s].Solution[param2change] -
                            self.Foods[neighbor].Solution[param2change])
                    Solution.repair()
                    self.tryEval(Solution)
                    if Solution.Fitness < self.Foods[s].Fitness:
                        self.checkForBest(newSolution)
                        self.Foods[s] = Solution
                        self.Trial[s] = 0
                    else:
                        self.Trial[s] += 1
                s += 1
                if s == self.FoodNumber:
                    s = 0

            mi = self.Trial.index(max(self.Trial))
            if self.Trial[mi] >= self.Limit:
                self.Foods[mi] = SolutionABC(self.D, self.Lower, self.Upper)
                self.tryEval2(self.Foods[mi])
                self.Trial[mi] = 0

            #backward process
            
            self.gBestFitness_record[self.iters_count] = self.Best.Fitness
            self.iters_count += 1
            #print(self.iters_count)
            
        
        #print(self.gBestFitness_record)
        return self.Best.Fitness
