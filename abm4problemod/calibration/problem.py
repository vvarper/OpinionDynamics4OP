import copy
from collections import deque

import numpy as np
import pandas as pd
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

import abm4problemod.model
from abm4problemod.runner import mc_run


class ODABMCalibrationProblem(FloatProblem):
    def __init__(self, model: abm4problemod.model.BaseODModel,
                 fixed_parameters: dict, history, measure_time: list,
                 metrics: dict, mc: int, dynamic_local: bool = False,
                 statistic: str = 'Concern',
                 synth: bool = False,
                 num_processes: int = 1,
                 generator: np.random.Generator = np.random.default_rng(),
                 polarization: bool = True, algorithmic: bool = False,
                 constrained: bool = True):

        super(ODABMCalibrationProblem, self).__init__()

        self.model = model
        self.polarization = polarization
        self.algorithmic = algorithmic
        self.fixed_parameters = {**fixed_parameters}

        self.mc = mc
        self.synth = synth
        self.num_processes = num_processes
        self.measure_time = measure_time
        self.num_months = len(self.measure_time) - 1

        self.history = history
        self.metrics = metrics
        self.statistic = statistic
        self.constrained = constrained

        if statistic == 'AvgOpinion':
            self.fixed_parameters['collector_statistic'] = 'avg_opinions'
        elif statistic == 'Concern':
            self.fixed_parameters['collector_statistic'] = 'concern'
        else:
            raise ValueError(f'Statistic {statistic} not supported.')

        self.dynamic_local = dynamic_local

        self.calib_parameters, self.lower_bound, self.upper_bound = [], [], []
        self.paired_variables = None

        # Process Local OD Parameters
        if self.model.__name__ == 'FJModel':
            self.calib_parameters = ['susceptibility']
            self.lower_bound = [0.1]
            self.upper_bound = [1.0]
        elif self.model.__name__ == 'BiasedAssimilationModel':
            self.calib_parameters = ['bias']
            self.lower_bound = [0.1]
            self.upper_bound = [100.0]
        else:
            self.calib_parameters = ['convergence', 'threshold_bc']
            self.lower_bound = [0.01, 0.0]
            if constrained:
                self.upper_bound = [0.5, 0.5]
            else:
                self.upper_bound = [0.5, 1.0]
            if polarization:
                self.calib_parameters += ['threshold_pol']
                if constrained:
                    self.lower_bound += [0.5]
                else:
                    self.lower_bound += [0.0]
                self.upper_bound += [1.0]
                self.paired_variables = tuple([(1, 2)])
            if algorithmic:
                self.calib_parameters += ['gamma']
                self.lower_bound += [0.1]
                self.upper_bound += [2.0]

        # Transform local parameters to dynamic
        if self.dynamic_local:
            self.calib_parameters = [f'{param}_month_{i}' for param in
                                     self.calib_parameters for i in
                                     range(self.num_months)]
            self.lower_bound = [value for value in self.lower_bound for _ in
                                range(self.num_months)]
            self.upper_bound = [value for value in self.upper_bound for _ in
                                range(self.num_months)]
            if self.model.__name__ == 'ATBCRModel' and polarization:
                self.paired_variables = tuple(
                    [(self.num_months + i, 2 * self.num_months + i) for i in
                     range(self.num_months)])

        self.number_of_variables = len(self.calib_parameters)
        self.random = generator

    def number_of_variables(self) -> int:
        return self.number_of_variables

    def number_of_objectives(self) -> int:
        return len(self.metrics)

    def number_of_constraints(self) -> int:
        return 0

    def idx_paired_variables(self) -> tuple:
        return self.paired_variables

    def is_valid_solution(self, solution: FloatSolution) -> bool:
        return self.is_valid(solution.variables)

    def is_valid(self, variables: list) -> bool:
        if (self.constrained and self.model == abm4problemod.model.ATBCRModel
                and self.polarization):
            for i, j in self.paired_variables:
                if abs(variables[i] - variables[j]) > 0.9 or abs(
                        variables[i] - variables[j]) < 0.1:
                    return False

        return True

    def create_solution_empty(self) -> FloatSolution:
        new_solution = FloatSolution(self.lower_bound, self.upper_bound,
                                     self.number_of_objectives(),
                                     self.number_of_constraints())

        return new_solution

    def create_solution_from_values(self, values: list) -> FloatSolution:
        new_solution = self.create_solution()
        new_solution.variables = values

        return new_solution

    def create_solution(self) -> FloatSolution:
        new_solution = self.create_solution_empty()

        is_valid = False
        while not is_valid:
            new_solution.variables = self.random.uniform(self.lower_bound,
                                                         self.upper_bound)

            is_valid = self.is_valid(new_solution.variables)

        return new_solution

    def name(self) -> str:
        return 'ODABMCalibrationProblem'

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives = self.evaluate_variables(solution.variables)

        return solution

    def evaluate_variables(self, values: list) -> list:
        parameters = self.decode_variables(values)

        results = mc_run(model_cls=self.model, parameters=parameters,
                         mc=self.mc, synth=self.synth,
                         number_processes=self.num_processes,
                         data_collection_period=1)

        return self.compute_all_fitness(results.raw_global_results)

    def decode_variables(self, variables: list):
        parameters = copy.deepcopy(self.fixed_parameters)
        factor = self.num_months if self.dynamic_local else 1

        # Process Local OD Parameters
        if self.model.__name__ == 'ATBCRModel':
            parameters['convergence'] = deque(
                [(self.measure_time[i], variables[i]) for i in range(factor)])
            parameters['threshold_bc'] = deque(
                [(self.measure_time[i], variables[factor + i]) for i in
                 range(factor)])
            if self.polarization:
                if self.constrained:
                    parameters['threshold_pol'] = deque(
                        [(self.measure_time[i], variables[factor * 2 + i]) for
                         i in
                         range(factor)])
                else:
                    parameters['threshold_pol'] = deque(
                        [(self.measure_time[i],
                          variables[factor + i] + (1 - variables[factor + i]) *
                          variables[factor * 2 + i]) for i in range(factor)])
            else:
                parameters['threshold_pol'] = deque(
                    [(self.measure_time[i], 1.0) for i in range(factor)])
                if self.algorithmic:
                    parameters['gamma'] = deque(
                        [(self.measure_time[i], variables[factor * 2 + i]) for
                         i in range(factor)])
        elif self.model.__name__ == 'FJModel':
            parameters['susceptibility'] = deque(
                [(self.measure_time[i], variables[i]) for i in range(factor)])

        else:
            parameters['bias'] = deque(
                [(self.measure_time[i], variables[i]) for i in range(factor)])

        return parameters

    def compute_fitness(self, results: pd.DataFrame, metric: callable):
        is_valid = np.isfinite(self.history)

        fitness = np.zeros(self.mc)
        for i in range(self.mc):
            output = \
                results.loc[results['Seed'] == i][self.statistic].to_numpy()[
                    self.measure_time]

            fitness[i] = metric(self.history[is_valid], output[is_valid])

        return fitness.mean()

    def compute_all_fitness(self, results: pd.DataFrame):

        fitness = []
        for metric in self.metrics.keys():
            fitness.append(self.compute_fitness(results, self.metrics[metric]))

        return tuple(fitness)
