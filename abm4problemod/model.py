import json
import os
from collections import deque

import mesa
import networkx as nx
import numpy as np

from abm4problemod import agent, scheduler


def compute_concern(model):
    return model.num_concerned / model.num_agents


def compute_avg_opinion(model):
    return model.sum_opinions / model.num_agents


def compute_std_opinion(model):
    n = model.num_agents
    mean_opinion = compute_avg_opinion(model)
    variance_opinion = (model.sum_opinions_sq - n * mean_opinion ** 2) / n
    return variance_opinion ** 0.5


def get_random_opinion(model):
    return model.schedule.agents[
        model.random_agents_report[model._steps]].opinion


class BaseODModel(mesa.Model):
    def __init__(self, num_agents, initial_op, simulation_steps, seed,
                 edge_list, concern_threshold=0.9,
                 collector_statistic='concern',
                 collector_full=False, evolution_params=None):
        super().__init__()

        self.evolution_params = {}
        self.evolution_transitions = {}
        if evolution_params:
            for param in evolution_params:
                self.evolution_params[param] = \
                    evolution_params[param].popleft()[1]
            self.evolution_transitions = evolution_params

        self.min_opinion = 0.0
        self.max_opinion = 1.0
        self.num_agents = num_agents

        self.simulation_steps = simulation_steps
        self.concern_threshold = concern_threshold
        self.collector_full = collector_full
        self.collector_statistic = collector_statistic

        if collector_statistic == 'concern':
            self.num_concerned = sum(
                [1 for op in initial_op if op > concern_threshold])
        elif collector_statistic == 'avg_opinions':
            self.sum_opinions = sum(initial_op)
        elif collector_statistic == 'avgstd_opinions':
            self.sum_opinions = sum(initial_op)
            self.sum_opinions_sq = sum([op ** 2 for op in initial_op])

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_agents))
        self.graph.add_edges_from(edge_list)

        self.seed = seed
        self.random = np.random.default_rng(seed)
        self.seeding = self.random.permutation(self.num_agents)

        self._steps = 0
        self.running = True

    def _set_reporters(self):

        model_reporters = {}

        if self.collector_statistic == 'concern':
            model_reporters["Concern"] = compute_concern
        elif self.collector_statistic == 'avg_opinions':
            model_reporters["AvgOpinion"] = compute_avg_opinion
        elif self.collector_statistic == 'avgstd_opinions':
            model_reporters["AvgOpinion"] = compute_avg_opinion
            model_reporters["StdOpinion"] = compute_std_opinion

        # At each step, report the state of a random agent
        if self.collector_full:
            # Random generator who not change the model state
            generator = np.random.default_rng(self.seed)
            self.random_agents_report = generator.choice(self.num_agents,
                                                         self.simulation_steps + 2)
            model_reporters["Opinion"] = get_random_opinion

            # Save in a JSON file the opinions of all agents in 100 equally spaced steps
            self.agents_results_file = (f"outputs/{self._get_config_name()}"
                                        f"/agent_opinions_{self.seed}.json")
            jump = (self.simulation_steps + 1) // 100
            self.agent_reporter_period = jump if jump > 0 else 1

            os.makedirs(os.path.dirname(self.agents_results_file),
                        exist_ok=True)

            with open(self.agents_results_file, 'w') as f:
                f.write('[')

        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters)

    def _get_config_name(self):
        return (
            f"{self.__class__.__name__}_{self.concern_threshold:.2}_{self.num_agents}")

    def __store_agents_opinions(self):
        data = {'Step': self._steps,
                'Opinions': [a.opinion for a in self.schedule.agents]}

        json_data = json.dumps(data)

        with open(self.agents_results_file, 'a') as f:
            if os.stat(self.agents_results_file).st_size > 1:
                f.write(',\n')
            f.write(json_data)

            if not self.running:
                f.write(']')

    def _update_agent_evolution_params(self):
        for param in self.evolution_params:
            if self.evolution_transitions[param] and self._steps == \
                    self.evolution_transitions[param][0][0]:
                self.evolution_params[param] = \
                    self.evolution_transitions[param].popleft()[1]

    def finish(self):
        return self._steps > self.simulation_steps

    def step(self):

        self._update_agent_evolution_params()

        self.schedule.step()
        self._steps += 1

        self.running = not self.finish()
        self.collect_data()

    def collect_data(self):
        if self.collector_full and (
                self._steps % self.agent_reporter_period == 0 or not self.running):
            self.__store_agents_opinions()

        self.datacollector.collect(self)

    def run_model(self):
        while self.running:
            self.step()


class FJModel(BaseODModel):
    def __init__(self, num_agents, initial_op, edge_list, simulation_steps,
                 seed, susceptibility, concern_threshold=0.9,
                 collector_statistic='concern', collector_full=False):
        evolution_params = {'susceptibility': susceptibility}

        super().__init__(num_agents, initial_op, simulation_steps, seed,
                         edge_list, concern_threshold, collector_statistic,
                         collector_full, evolution_params)

        self.schedule = scheduler.NeighborsScheduler(self, initial_op,
                                                     agent.FJAgent)

        self._set_reporters()
        self.collect_data()

    def _get_config_name(self):
        config = super()._get_config_name()
        return f"{config}_{self.evolution_params['susceptibility']:.2}"


class BiasedAssimilationModel(BaseODModel):
    def __init__(self, num_agents, initial_op, edge_list, simulation_steps,
                 seed, bias, concern_threshold=0.9,
                 collector_statistic='concern', collector_full=False):
        evolution_params = {'bias': bias}

        super().__init__(num_agents, initial_op, simulation_steps, seed,
                         edge_list, concern_threshold, collector_statistic,
                         collector_full, evolution_params)

        self.schedule = scheduler.NeighborsScheduler(self, initial_op,
                                                     agent.BiasedAssimilationAgent)

        self._set_reporters()
        self.collect_data()

    def _get_config_name(self):
        config = super()._get_config_name()
        return f"{config}_{self.evolution_params['bias']:.2}"


class ATBCRModel(BaseODModel):
    def __init__(self, num_agents, initial_op, edge_list, simulation_steps,
                 seed, threshold_bc, threshold_pol,
                 convergence=deque([(0, 0.1)]), gamma=None,
                 concern_threshold=0.9, collector_statistic='concern',
                 collector_full=False):

        evolution_params = {'threshold_bc': threshold_bc,
                            'threshold_pol': threshold_pol,
                            'convergence': convergence,
                            'gamma': gamma if gamma is not None else deque(
                                [(0, 0.0)])}

        super().__init__(num_agents, initial_op, simulation_steps, seed,
                         edge_list, concern_threshold, collector_statistic,
                         collector_full, evolution_params)

        if gamma is None:
            self.schedule = scheduler.PairwiseRandomScheduler(self, initial_op,
                                                              agent.ATBCRAgent)
        else:
            self.schedule = scheduler.PairwiseAlgorithmicBiasedScheduler(self,
                                                                         initial_op,
                                                                         agent.ATBCRAgent)

        self._set_reporters()
        self.collect_data()

    def _get_config_name(self):
        config = super()._get_config_name()

        return (f"{config}_{self.evolution_params['threshold_bc']:.2}"
                f"_{self.evolution_params['threshold_pol']:.2}"
                f"_{self.evolution_params['gamma']:.2}")
