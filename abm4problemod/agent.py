import mesa


class BaseODAgent(mesa.Agent):
    def __init__(self, unique_id, model, opinion):
        super().__init__(unique_id, model)
        self.opinion = opinion

    def register_concern_change(self, old_opinion):
        if old_opinion < self.model.concern_threshold <= self.opinion:
            self.model.num_concerned += 1
        elif old_opinion >= self.model.concern_threshold > self.opinion:
            self.model.num_concerned -= 1

    def update_opinion_stats(self, old_opinion):
        diff_opinion = self.opinion - old_opinion
        self.model.sum_opinions += diff_opinion
        self.model.sum_opinions_sq += (
                                              self.opinion + old_opinion) * diff_opinion


class ATBCRAgent(BaseODAgent):
    def talk(self, difference):
        old_opinion = self.opinion

        if abs(difference) < self.model.evolution_params['threshold_bc']:
            self.opinion += self.model.evolution_params[
                                'convergence'] * difference
        elif abs(difference) > self.model.evolution_params['threshold_pol']:
            self.opinion -= self.model.evolution_params[
                                'convergence'] * difference

        self.opinion = max(self.model.min_opinion,
                           min(self.opinion, self.model.max_opinion))

        if self.model.collector_statistic == 'concern':
            self.register_concern_change(old_opinion)
        elif self.model.collector_statistic == 'avg_opinions':
            self.model.sum_opinions += (self.opinion - old_opinion)
        elif self.model.collector_statistic == 'avgstd_opinions':
            self.update_opinion_stats(old_opinion)


class FJAgent(BaseODAgent):
    def __init__(self, unique_id, model, opinion):
        super().__init__(unique_id, model, opinion)
        self.initial_opinion = opinion

    def talk(self):
        old_opinion = self.opinion
        susceptibility = self.model.evolution_params['susceptibility']
        neighbors = list(self.model.graph.neighbors(self.unique_id)) + [
            self.unique_id]

        new_opinion = sum(
            [self.model.schedule.agents[neighbor].opinion for neighbor in
             neighbors]) / (len(neighbors))

        self.opinion = susceptibility * new_opinion + (
                1 - susceptibility) * self.initial_opinion

        if self.model.collector_statistic == 'concern':
            self.register_concern_change(old_opinion)
        elif self.model.collector_statistic == 'avg_opinions':
            self.model.sum_opinions += (self.opinion - old_opinion)
        elif self.model.collector_statistic == 'avgstd_opinions':
            self.update_opinion_stats(old_opinion)


class BiasedAssimilationAgent(BaseODAgent):
    def __init__(self, unique_id, model, opinion):
        super().__init__(unique_id, model, opinion)
        self.initial_opinion = opinion

    def talk(self):
        old_opinion = self.opinion
        bias = self.model.evolution_params['bias']
        neighbors = list(self.model.graph.neighbors(self.unique_id))

        neighbors_sum = sum(
            [self.model.schedule.agents[neighbor].opinion for neighbor in
             neighbors])
        neighbors_sum_biased = neighbors_sum * (old_opinion ** bias)

        denominator = 1 + neighbors_sum_biased + (
                len(neighbors) - neighbors_sum) * (
                              (1 - old_opinion) ** bias)

        self.opinion = (old_opinion + neighbors_sum_biased) / denominator

        if self.model.collector_statistic == 'concern':
            self.register_concern_change(old_opinion)
        elif self.model.collector_statistic == 'avg_opinions':
            self.model.sum_opinions += (self.opinion - old_opinion)
        elif self.model.collector_statistic == 'avgstd_opinions':
            self.update_opinion_stats(old_opinion)
