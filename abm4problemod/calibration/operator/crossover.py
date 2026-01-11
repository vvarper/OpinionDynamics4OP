import copy
from typing import List

import numpy as np
from jmetal.core.solution import FloatSolution
from jmetal.operator import DifferentialEvolutionCrossover


class GeneralizedDifferentialEvolutionCrossover(
    DifferentialEvolutionCrossover):
    def __init__(self, max_diff: float = None, min_diff: float = None,
                 comparing_idx: tuple = None,
                 CR: float = None, F: float = None, crossover: str = "rand",
                 shade_clip: bool = False,
                 generator: np.random.Generator = np.random.default_rng()):
        super(GeneralizedDifferentialEvolutionCrossover, self).__init__(
            CR, F)

        self.current_individual = None

        self.max_diff = max_diff
        self.min_diff = min_diff
        self.comparing_idx = comparing_idx
        self.crossover = crossover
        self.shade_clip = shade_clip
        self.random = generator

    def adjust_value_to_bounds(self, value, lower_bound, upper_bound,
                               current_value):
        if value < lower_bound:
            return (
                           current_value + lower_bound) / 2 if self.shade_clip else lower_bound
        if value > upper_bound:
            return (
                           current_value + upper_bound) / 2 if self.shade_clip else upper_bound
        return value

    def enforce_difference_constraints(self, child):
        for idx1, idx2 in self.comparing_idx:
            if idx1 >= len(child.variables) or idx2 >= len(child.variables):
                raise ValueError("Index out of range for comparing_idx.")

            diff = abs(child.variables[idx2] - child.variables[idx1])
            adjustment = 0
            if diff > self.max_diff:
                adjustment = (diff - self.max_diff) / 2
            elif diff < self.min_diff:
                adjustment = (self.min_diff - diff) / 2

            if adjustment:
                if child.variables[idx2] > child.variables[idx1]:
                    child.variables[idx1] += adjustment
                    child.variables[idx2] -= adjustment
                else:
                    child.variables[idx1] -= adjustment
                    child.variables[idx2] += adjustment

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if len(parents) != self.get_number_of_parents():
            raise Exception("The number of parents is not {}: {}".format(
                self.get_number_of_parents(), len(parents)))

        if self.current_individual is None:
            raise ValueError(
                "current_individual must be set before executing the crossover.")

        child = copy.deepcopy(self.current_individual)
        number_of_variables = len(parents[0].variables)
        rand = self.random.integers(0, number_of_variables)

        for i in range(number_of_variables):
            if self.random.random() < self.CR or i == rand:
                if self.crossover == "rand":
                    value = parents[2].variables[i] + self.F * (
                            parents[0].variables[i] - parents[1].variables[
                        i])
                else:
                    value = child.variables[i] + self.F * (
                            parents[2].variables[i] - child.variables[
                        i]) + self.F * (parents[0].variables[i] -
                                        parents[1].variables[i])

                value = self.adjust_value_to_bounds(value,
                                                    child.lower_bound[i],
                                                    child.upper_bound[i],
                                                    child.variables[i])
            else:
                value = child.variables[i]

            child.variables[i] = value

        if self.comparing_idx:
            self.enforce_difference_constraints(child)

        return [child]

    def get_name(self) -> str:
        return "Generalized Differential Evolution crossover"
