import numpy as np

from abm4problemod import agent


class BaseODScheduler:
    def __init__(self, model, initial_op, agent_class=agent.BaseODAgent):
        self.model = model
        self.agents = [agent_class(i, model, initial_op[j]) for
                       i, j in enumerate(model.seeding)]
        self.steps = 0


class NeighborsScheduler(BaseODScheduler):
    def __init__(self, model, initial_op, agent_class):
        super().__init__(model, initial_op, agent_class)
        self.choices = self.model.random.choice(len(initial_op),
                                                self.model.simulation_steps + 1)

    def step(self):
        id_ag = self.choices[self.steps]
        self.agents[id_ag].talk()

        self.steps += 1


class PairwiseRandomScheduler(BaseODScheduler):
    def __init__(self, model, initial_op, agent_class):
        super().__init__(model, initial_op, agent_class)

        edge_list = list(self.model.graph.edges)

        choices = self.model.random.choice(len(edge_list),
                                           self.model.simulation_steps + 1)
        self.choices = [edge_list[i] for i in choices]

    def step(self):
        id_ag1, id_ag2 = self.choices[self.steps]
        ag1_op = self.agents[id_ag1].opinion
        ag2_op = self.agents[id_ag2].opinion

        self.agents[id_ag1].talk(ag2_op - ag1_op)
        self.agents[id_ag2].talk(ag1_op - ag2_op)

        self.steps += 1


class PairwiseAlgorithmicBiasedScheduler(BaseODScheduler):
    def __init__(self, model, initial_op, agent_class):
        super().__init__(model, initial_op, agent_class)
        self.choices = self.model.random.choice(len(initial_op),
                                                self.model.simulation_steps + 1)

        self.random_for_pairs = self.model.random.random(
            self.model.simulation_steps + 1)

    def step(self):
        id_ag1 = self.choices[self.steps]

        # Select id_ag2 as a neighbor based on the opinion distance and gamma parameter
        # 1. Get the neighbors of id_ag1 and their opinions
        neighbors = list(
            self.model.graph.neighbors(self.agents[id_ag1].unique_id))
        neighbors_op = np.array([self.agents[n].opinion for n in neighbors])

        # 2. Calculate the selection probability based on the distance to id_ag1's opinion
        dist = np.clip(np.abs(neighbors_op - self.agents[id_ag1].opinion),
                       0.00001, None)
        weights = np.power(dist, -self.model.evolution_params['gamma'])
        weights /= np.sum(weights)
        cumulative_weights = np.cumsum(weights)

        # 4. Select id_ag2 based on the cumulative selection probability
        n2 = np.argmax(cumulative_weights >= self.random_for_pairs[self.steps])
        id_ag2 = neighbors[n2]

        # Talk between the two agents
        ag1_op = self.agents[id_ag1].opinion
        ag2_op = self.agents[id_ag2].opinion

        self.agents[id_ag1].talk(ag2_op - ag1_op)
        self.agents[id_ag2].talk(ag1_op - ag2_op)

        self.steps += 1
