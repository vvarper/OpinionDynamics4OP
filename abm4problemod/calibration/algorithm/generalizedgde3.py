from typing import Generator, Type

from jmetal.algorithm.multiobjective import GDE3
from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.comparator import Comparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion


class GeneralizedGDE3(GDE3):
    def __init__(
            self,
            problem: Problem,
            population_size: int,
            cr: float,
            f: float,
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            k: float = 0.5,
            population_generator: Generator = store.default_generator,
            population_evaluator: Evaluator = store.default_evaluator,
            dominance_comparator: Comparator = store.default_comparator,
            crossover_operator: Type[
                DifferentialEvolutionCrossover] = DifferentialEvolutionCrossover,
            selection_operator: Type[
                DifferentialEvolutionSelection] = DifferentialEvolutionSelection,
            crossover_kwargs: dict = None
    ):
        super(GeneralizedGDE3, self).__init__(
            problem=problem,
            population_size=population_size,
            cr=cr,
            f=f,
            termination_criterion=termination_criterion,
            k=k,
            population_generator=population_generator,
            population_evaluator=population_evaluator,
            dominance_comparator=dominance_comparator,
        )

        self.crossover_operator = crossover_operator(CR=cr, F=f, **(
                crossover_kwargs or {}))

        self.selection_operator = selection_operator()
