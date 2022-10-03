import random as rnd
import copy
from NiaPy.benchmarks.utility import Utility
import numpy 

__all__ = ['DifferentialEvolutionAlgorithm']


class SolutionDE:

    def __init__(self, D, LB, UB):
        self.D = D
        self.LB = LB
        self.UB = UB

        self.Solution = []
        self.Fitness = float('inf')
        self.generateSolution()

    def generateSolution(self):
        self.Solution = [self.LB[i] + (self.UB[i] - self.LB[i]) * rnd.random()
                         for i in range(self.D)]

    def evaluate(self):
        self.Fitness = SolutionDE.FuncEval(self.D, self.Solution)

    def repair(self):
        for i in range(self.D):
            if self.Solution[i] > self.UB[i]:
                self.Solution[i] = self.UB[i]
            if self.Solution[i] < self.LB[i]:
                self.Solution[i] = self.LB[i]

    def __eq__(self, other):
        return self.Solution == other.Solution and self.Fitness == other.Fitness


class DifferentialEvolutionAlgorithm:
    r"""Implementation of Differential evolution algorithm.

    **Algorithm:** Differential evolution algorithm

    **Date:** 2018

    **Author:** Uros Mlakar

    **License:** MIT

    **Reference paper:**
        Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and
        efficient heuristic for global optimization over continuous spaces."
        Journal of global optimization 11.4 (1997): 341-359.
    """

    def __init__(self, D, NP, nFES, F, CR, benchmark):
        r"""**__init__(self, D, NP, nFES, F, CR, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            F {decimal} -- scaling factor

            CR {decimal} -- crossover rate

            benchmark {object} -- benchmark implementation object

        Raises:
            TypeError -- Raised when given benchmark function which does not exists.

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.D = D  # dimension of problem
        self.Np = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.F = F  # scaling factor
        self.CR = CR  # crossover rate
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        
        #modified bound
        self.Lower_D = self.benchmark.Lower_D  # lower bound
        self.Upper_D = self.benchmark.Upper_D  # upper bound

        SolutionDE.FuncEval = staticmethod(self.benchmark.function())
        self.Population = []
        self.bestSolution = SolutionDE(self.D, self.Lower_D, self.Upper_D)

        #record the best solution for each iter
        self.iters_count = 0
        self.iters = int(self.nFES/self.Np)
        self.gBestFitness_record = numpy.zeros([self.iters])

    def evalPopulation(self):
        """Evaluate population."""

        for p in self.Population:
            p.evaluate()
            if p.Fitness < self.bestSolution.Fitness:
                self.bestSolution = copy.deepcopy(p)

    def initPopulation(self):
        """Initialize population."""

        for _i in range(self.Np):
            self.Population.append(SolutionDE(self.D, self.Lower_D, self.Upper_D))

    def generationStep(self, Population):
        """Implement main generation step."""

        newPopulation = []
        for i in range(self.Np):
            newSolution = SolutionDE(self.D, self.Lower_D, self.Upper_D)

            r = rnd.sample(range(0, self.Np), 3)
            while i in r:
                r = rnd.sample(range(0, self.Np), 3)
            jrand = int(rnd.random() * self.Np)

            for j in range(self.D):
                if rnd.random() < self.CR or j == jrand:
                    newSolution.Solution[j] = Population[r[0]].Solution[j] + self.F * \
                        (Population[r[1]].Solution[j] - Population[r[2]].Solution[j])
                else:
                    newSolution.Solution[j] = Population[i].Solution[j]
            newSolution.repair()
            newSolution.evaluate()

            if newSolution.Fitness < self.bestSolution.Fitness:
                self.bestSolution = copy.deepcopy(newSolution)
            if newSolution.Fitness < self.Population[i].Fitness:
                newPopulation.append(newSolution)
            else:
                newPopulation.append(Population[i])
        return newPopulation

    def run(self):
        """Run."""
        self.initPopulation()
        self.evalPopulation()
        FEs = self.Np
        while FEs <= self.nFES:
            #print(self.iters_count)
            #print(self.bestSolution.Fitness)
            self.gBestFitness_record[self.iters_count] = self.bestSolution.Fitness
            self.iters_count+=1
            self.Population = self.generationStep(self.Population)
            FEs += self.Np
            
        return self.bestSolution.Fitness
