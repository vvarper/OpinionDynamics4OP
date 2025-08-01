import copy
from typing import List, TypeVar

import numpy as np
from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import StoppingByEvaluations
from scipy.stats import cauchy

from abm4problemod.calibration.operator.crossover import \
    GeneralizedDifferentialEvolutionCrossover
from abm4problemod.calibration.operator.selection import \
    CurrentToPBestDifferentialEvolutionSelection

S = TypeVar("S")
R = TypeVar("R")


class LSHADE(EvolutionaryAlgorithm[S, R]):
    def __init__(
            self,
            problem: Problem,
            population_size: int,
            max_evaluations: int,
            population_generator: Generator = store.default_generator,
            population_evaluator: Evaluator = store.default_evaluator,
            objective_idx: int = 0,
            crossover_kwargs: dict = None,
            H: int = 6,
            reduce_population: bool = False,
            use_archive: bool = False,
            generator: np.random.Generator = np.random.default_rng()
    ):
        super(LSHADE, self).__init__(
            problem=problem, population_size=population_size,
            offspring_population_size=population_size
        )

        self.termination_criterion = StoppingByEvaluations(max_evaluations)
        self.observable.register(self.termination_criterion)
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.obj_idx = objective_idx
        self.selection_operator = CurrentToPBestDifferentialEvolutionSelection(
            generator=generator)
        self.crossover_operator = GeneralizedDifferentialEvolutionCrossover(
            crossover="current-to-best", shade_clip=True, generator=generator,
            **(crossover_kwargs or {}))

        # Population sizes for linear reduction
        self.max_population_size = population_size
        self.min_population_size = 4 if reduce_population else 0

        # Memory for control parameters
        self.H = H  # Memory size
        self.m_F = [0.5] * H  # Memory for Scaling factor
        self.m_CR = [0.5] * H  # Memory for Crossover rate
        self.k = 0  # Index for control parameters memory
        self.F = []  # Current scaling factor by solution
        self.CR = []  # Current crossover rate by solution

        # Archive for historical solutions
        self.archive = []
        self.archive_size = population_size if use_archive else 0

        self.random = generator

    def init_progress(self) -> None:
        super().init_progress()
        self.solutions.sort(key=lambda x: x.objectives[self.obj_idx])

    def step(self):
        self.select_control_parameters()

        mating_pool = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_pool)
        self.evaluate(offspring_population)

        new_solutions = self.replacement(self.solutions,
                                         offspring_population)

        self.update_memories(self.solutions, offspring_population)

        self.solutions = new_solutions
        self.solutions.sort(key=lambda x: x.objectives[self.obj_idx])

        if self.min_population_size:
            self.reduce_population()

    def select_control_parameters(self):
        self.CR = []
        self.F = []
        r = self.random.integers(0, self.H, self.population_size)

        for i in range(self.population_size):

            # Generate random normal value with numpy
            CR = np.clip(self.random.normal(loc=self.m_CR[r[i]], scale=0.1),
                         0, 1)

            F = 0
            while F <= 0:
                # Use self.random to generate a cauchy random value
                F = np.clip(cauchy.rvs(loc=self.m_F[r[i]], scale=0.1,
                                       random_state=self.random), 0, 1)

            self.CR.append(CR)
            self.F.append(F)

    def selection(self, mating_pool: List[S]) -> List[S]:
        mating_pool = []
        self.selection_operator.set_pmin(self.solutions)
        self.selection_operator.set_archive(self.archive)

        for i in range(self.population_size):
            self.selection_operator.set_current_index(i)
            selected_solutions = self.selection_operator.execute(
                self.solutions)
            mating_pool = mating_pool + selected_solutions

        return mating_pool

    def reproduction(self, mating_pool: List[S]) -> List[S]:
        offspring_population = []

        for i in range(self.population_size):
            self.crossover_operator.current_individual = self.solutions[i]
            self.crossover_operator.F = self.F[i]
            self.crossover_operator.CR = self.CR[i]
            parents = mating_pool[i * 3: (i + 1) * 3]

            offspring_population.append(
                self.crossover_operator.execute(parents)[0])

        return offspring_population

    def replacement(self, population: List[S],
                    offspring_population: List[S]) -> List[S]:

        new_population = []

        for solution1, solution2 in zip(population, offspring_population):
            if solution1.objectives[self.obj_idx] < solution2.objectives[
                self.obj_idx]:
                new_population.append(solution1)
            else:
                new_population.append(solution2)

        return new_population

    def update_memories(self, parents: List[S], offspring: List[S]):
        s_CR = []
        s_F = []
        weights = []

        for i in range(self.population_size):
            if offspring[i].objectives[self.obj_idx] < parents[i].objectives[
                self.obj_idx]:
                if self.archive_size:
                    self.archive.append(copy.deepcopy(parents[i]))
                s_CR.append(self.CR[i])
                s_F.append(self.F[i])
                weights.append(parents[i].objectives[self.obj_idx] -
                               offspring[i].objectives[self.obj_idx])

        if len(s_CR) > 0:
            total_weights = sum(weights)
            weights = [w / total_weights for w in weights]
            self.m_CR[self.k] = sum(w * cr for w, cr in zip(weights, s_CR))
            total_F = sum(w * f for w, f in zip(weights, s_F))
            self.m_F[self.k] = sum(
                w * f * f for w, f in zip(weights, s_F)) / total_F
            self.k = (self.k + 1) % self.H

        if self.archive_size and len(self.archive) > self.archive_size:
            indexes = np.random.permutation(len(self.archive))[
                      :self.archive_size]
            self.archive = [self.archive[i] for i in indexes]

    def reduce_population(self):
        self.population_size = round(
            ((self.min_population_size - self.max_population_size)
             / self.termination_criterion.max_evaluations)
            * (
                    self.evaluations + self.population_size) + self.max_population_size)

        self.offspring_population_size = self.population_size

        self.solutions = self.solutions[:self.population_size]

    def create_initial_solutions(self) -> List[S]:
        return [self.population_generator.new(self.problem) for _ in
                range(self.population_size)]

    def evaluate(self, population: List[S]):
        return self.population_evaluator.evaluate(population, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "L-SHADE"
