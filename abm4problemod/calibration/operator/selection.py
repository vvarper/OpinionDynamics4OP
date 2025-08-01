from typing import List, TypeVar

import numpy as np
from jmetal.core.operator import Selection
from jmetal.core.solution import Solution

S = TypeVar("S", bound=Solution)


class CurrentToPBestDifferentialEvolutionSelection(
    Selection[List[S], List[S]]):
    def __init__(self,
                 generator: np.random.Generator = np.random.default_rng()):
        super(CurrentToPBestDifferentialEvolutionSelection, self).__init__()
        self.current_index = None
        self.pmin = None
        self.memory = None
        self.random = generator

    def set_pmin(self, solutions: List[S]):
        self.pmin = 2.0 / len(solutions)

    def set_archive(self, archive: List[S]):
        self.memory = archive or []

    def set_current_index(self, index: int):
        self.current_index = index

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        elif len(front) < 4:
            raise Exception(
                "The front has less than four solutions: " + str(len(front)))

        # Select r1 (not current_index)
        candidates = list(range(len(front)))
        candidates.remove(self.current_index)

        r1 = self.random.choice(candidates)

        # Select r2 (from front + memory, not current_index or r1)
        total_solutions = front + self.memory
        candidates = list(range(len(total_solutions)))
        candidates.remove(self.current_index)
        candidates.remove(r1)
        r2 = self.random.choice(candidates)

        p = self.pmin if self.pmin < 0.2 else self.random.uniform(0.2,
                                                                  self.pmin)
        num_best = max(1, int(p * len(front)))

        pbest = self.random.integers(0, num_best)

        return [front[r1], total_solutions[r2], front[pbest]]

    def get_name(self) -> str:
        return "Current-to-pbest Differential Evolution selection"
