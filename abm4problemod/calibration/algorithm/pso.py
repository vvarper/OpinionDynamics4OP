import copy
from typing import TypeVar, List

import numpy as np
from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import StoppingByEvaluations

S = TypeVar("S")
R = TypeVar("R")


class PSO(ParticleSwarmOptimization):
    def __init__(self, problem: Problem[S], swarm_size: int, max_evaluations,
                 max_diff: float,
                 min_diff: float,
                 comparing_idx: tuple,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator,
                 objective_idx: int = 0,
                 cognitive_weight: float = 1.49618,
                 social_weight: float = 1.49618,
                 inertia_weight: float = 0.7298,
                 max_speed_rate: float = 1.0,
                 generator: np.random.Generator = np.random.default_rng()):
        super(PSO, self).__init__(problem=problem, swarm_size=swarm_size)

        self.max_diff = max_diff
        self.min_diff = min_diff
        self.comparing_idx = comparing_idx

        self.termination_criterion = StoppingByEvaluations(max_evaluations)
        self.observable.register(self.termination_criterion)
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.obj_idx = objective_idx

        self.c1 = cognitive_weight
        self.c2 = social_weight
        self.w = inertia_weight
        self.local_best = []
        self.global_best = None
        self.speed = []

        self.max_speed = []
        for i in range(problem.number_of_variables):
            self.max_speed.append(max_speed_rate * (
                    problem.upper_bound[i] - problem.lower_bound[i]))

        self.random = generator

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        self.speed = [np.zeros(self.problem.number_of_variables).tolist() for _
                      in range(self.swarm_size)]

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            self.local_best.append(copy.deepcopy(swarm[i]))

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        self.global_best = min(swarm, key=lambda particle: particle.objectives[
            self.obj_idx])

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            r1 = self.random.uniform(size=self.problem.number_of_variables)
            r2 = self.random.uniform(size=self.problem.number_of_variables)

            for j in range(self.problem.number_of_variables):
                cognitive = self.c1 * r1[j] * (
                        self.local_best[i].variables[j] -
                        swarm[i].variables[j])
                social = self.c2 * r2[j] * (self.global_best.variables[j] -
                                            swarm[i].variables[j])

                self.speed[i][j] = self.w * self.speed[i][
                    j] + cognitive + social

                if self.speed[i][j] > self.max_speed[j]:
                    self.speed[i][j] = self.max_speed[j]
                if self.speed[i][j] < -self.max_speed[j]:
                    self.speed[i][j] = -self.max_speed[j]

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            if swarm[i].objectives[self.obj_idx] < \
                    self.local_best[i].objectives[
                        self.obj_idx]:
                self.local_best[i] = copy.deepcopy(swarm[i])

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            if swarm[i].objectives[self.obj_idx] < self.global_best.objectives[
                self.obj_idx]:
                self.global_best = copy.deepcopy(swarm[i])

    def update_position(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            for j in range(self.problem.number_of_variables):
                swarm[i].variables[j] += self.speed[i][j]

                if swarm[i].variables[j] < self.problem.lower_bound[j]:
                    swarm[i].variables[j] = self.problem.lower_bound[j]
                if swarm[i].variables[j] > self.problem.upper_bound[j]:
                    swarm[i].variables[j] = self.problem.upper_bound[j]

            if self.comparing_idx:
                self._enforce_difference_constraints(swarm[i])

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.swarm_generator.new(self.problem) for _ in
                range(self.swarm_size)]

    def evaluate(self, solution_list: List[FloatSolution]) -> List[
        FloatSolution]:
        return self.swarm_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def result(self) -> R:
        return [self.global_best]

    def get_name(self) -> str:
        return "PSO"

    def _enforce_difference_constraints(self, solution: FloatSolution) -> None:
        for idx1, idx2 in self.comparing_idx:
            if idx1 >= len(solution.variables) or idx2 >= len(
                    solution.variables):
                raise ValueError("Index out of range for comparing_idx.")

            diff = abs(solution.variables[idx2] - solution.variables[idx1])
            adjustment = 0
            if diff > self.max_diff:
                adjustment = (diff - self.max_diff) / 2
            elif diff < self.min_diff:
                adjustment = (self.min_diff - diff) / 2

            if adjustment:
                if solution.variables[idx2] > solution.variables[idx1]:
                    solution.variables[idx1] += adjustment
                    solution.variables[idx2] -= adjustment
                else:
                    solution.variables[idx1] -= adjustment
                    solution.variables[idx2] += adjustment
