import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def calculate_raw_results(results):
    processed_results = pd.DataFrame(results).sort_values(
        by=['Seed', 'Step']).reset_index(drop=True)

    return processed_results


def summarise_results(results):
    summary_df = results.drop(columns='Seed').groupby(['Step']).agg(
        ['mean', 'std', 'min', 'max']).reset_index()

    return summary_df


class SimulationStatistics:
    def __init__(self, results):
        self.raw_global_results = calculate_raw_results(results)
        self.summary_results = summarise_results(self.raw_global_results)

    def load_agents_results(self, sim_config):

        # 1. Get the file path
        file = f'outputs/{sim_config["model_name"]}_{sim_config["concern_threshold"]:.2}'

        if sim_config["model_name"] == 'ATBCRModel':
            file += (f'_{sim_config["threshold_bc"]:.2}_'
                     f'{sim_config["threshold_pol"]:.2}_{sim_config["gamma"]:.2}/')
        elif sim_config["model_name"] == 'FJModel':
            file += f'_{sim_config["susceptibility"]:.2}/'
        elif sim_config["model_name"] == 'BiasedAssimilationModel':
            file += f'_{sim_config["bias"]:.2}/'

        # 2. Load the results for each Monte Carlo run
        mc_results = []
        for i in range(max(1, sim_config['mc'])):
            with open(f'{file}agent_opinions_{i}.json') as f:
                results = json.load(f)
                for result in results:
                    result['Opinions'] = np.array(result['Opinions'])
                mc_results += results

        # 3. Group the results to get the mean opinions
        agents_opinions = pd.DataFrame(mc_results).groupby('Step')[
            'Opinions'].apply(
            lambda x: np.mean(np.stack(x), axis=0)).reset_index()

        return agents_opinions

    def plot_opinions(self, title, sim_config, filename="", alpha=0.1,
                      measure_time=None, xlabels=None):

        agents_opinions = self.load_agents_results(sim_config)

        initial_op = agents_opinions[
            agents_opinions['Step'] == agents_opinions['Step'].min()][
            'Opinions'].values[0]
        final_op = agents_opinions[
            agents_opinions['Step'] == agents_opinions['Step'].max()][
            'Opinions'].values[0]

        if sim_config["mc"] == -1:
            # Get opinions of self.raw_global_results['Seed'] == 0
            one_run_results = self.raw_global_results[
                self.raw_global_results['Seed'] == 0]
            intermediate_op = [one_run_results['Step'].to_list(),
                               one_run_results['Opinion'].to_list()]
        else:
            intermediate_op = [self.summary_results['Step'].to_list(),
                               self.summary_results['Opinion'][
                                   'mean'].to_list()]

        simulation_steps = self.summary_results['Step'].max()

        fig, ax = plt.subplots(1, 2, figsize=(6, 3),
                               gridspec_kw={'width_ratios': [4, 1]})
        fig.tight_layout()

        # SCATTER PLOT: left plot
        # Intermediate opinions (blue)
        ax[0].scatter(intermediate_op[0], intermediate_op[1], s=1,
                      color="blue", alpha=alpha, label="interm op.")
        # Initial opiniones (green)
        ax[0].scatter(np.full(len(initial_op), 0), initial_op, s=5,
                      color="green", alpha=alpha, label="initial op.")
        # Final opinions (red)
        ax[0].scatter(np.full(len(final_op), simulation_steps), final_op, s=5,
                      color="red", alpha=alpha, label="final op.")

        ax[0].set_xlim([-2 * simulation_steps // 100,
                        simulation_steps + 2 * simulation_steps // 100])
        ax[0].set_ylim([0, 1])
        ax[0].set_xlabel('time step')
        ax[0].set_ylabel('opinion')
        ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        if measure_time is not None:
            ax[0].set_xticks(measure_time)
            xlabels = pd.read_csv('data/simulation_periods.csv')['Subperiod']
            ax[0].set_xticklabels(xlabels, rotation=45)

        handles, labels = ax[0].get_legend_handles_labels()

        order = [1, 0, 2]
        leg = ax[0].legend([handles[idx] for idx in order],
                           [labels[idx] for idx in order], loc='center left',
                           bbox_to_anchor=(1.4, 0.85), fancybox=True)
        for lh in leg.legend_handles:
            lh.set_alpha(1)

        # HISTOGRAMS: right plot
        ax[1].hist(initial_op, bins=50, range=(0, 1), orientation="horizontal",
                   color="green", alpha=0.7)
        ax[1].hist(final_op, bins=50, range=(0, 1), orientation="horizontal",
                   color="red", alpha=0.7)

        ax[1].set_xscale("log")
        ax[1].set_ylim([0, 1])
        ax[1].set_xlim([len(final_op) // 100, len(final_op)])
        ax[1].set_ylabel('final opinion')
        ax[1].set_xlabel('%ag')
        ax[1].xaxis.tick_top()
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_label_position("right")
        ax[1].set_xticks(
            [len(final_op) // 100, len(final_op) // 10, len(final_op)])
        ax[1].set_xticklabels(["1%", "10%", "100%"])

        plt.subplots_adjust(wspace=0.03, hspace=0)
        ax[0].set_title(title)

        if not filename == "":
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

        plt.clf()

    def plot_concern(self, months=None, measure_time=None, history=None,
                     filename="", title=None):
        concern = self.summary_results['Concern']['mean'].to_list()

        # Mode with monthly data
        if measure_time is not None:
            if months is None:
                months = len(measure_time)
            concern = [concern[i] for i in measure_time][:months + 1]
            xlabels = pd.read_csv('data/simulation_periods.csv')['Subperiod']
            plt.xticks(range(0, len(history[:months + 1])),
                       xlabels[:months + 1],
                       rotation=45)

            if history is not None:
                plt.plot(history[:months + 1], color='blue', marker='.',
                         label='History')
        else:
            plt.xlabel('Time step')

        plt.plot(concern, marker='.', label=f'Simulation', color='green')

        plt.ylabel(f'Proportion of population concerned')
        plt.gca().set_yticklabels(
            ['{:.2f}%'.format(x * 100) for x in plt.gca().get_yticks()])
        plt.legend()

        if title is not None:
            plt.title(title)

        if not filename == "":
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

        plt.clf()

    def plot_avg_opinion(self, months=None, measure_time=None, filename=""):
        avg_opinion = self.summary_results['AvgOpinion']['mean'].to_list()
        std_opinion = self.summary_results['StdOpinion']['mean'].to_list()

        # Mode with monthly data
        if measure_time is not None:
            if months is None:
                months = len(measure_time)
            avg_opinion = [avg_opinion[i] for i in measure_time][:months + 1]
            std_opinion = [std_opinion[i] for i in measure_time][:months + 1]
            xlabels = pd.read_csv('data/simulation_periods.csv')['Subperiod']
            plt.xticks(range(0, len(avg_opinion)),
                       xlabels[:months + 1],
                       rotation=45)

        # Plot range given by the standard deviation
        plt.fill_between(range(len(avg_opinion)),
                         np.array(avg_opinion) - np.array(std_opinion),
                         np.array(avg_opinion) + np.array(std_opinion),
                         alpha=0.2)

        plt.plot(avg_opinion, marker='.', label='Simulation')

        plt.ylabel('Average opinion')
        plt.gca().set_yticklabels(
            ['{:.2f}'.format(x) for x in plt.gca().get_yticks()])
        plt.legend()

        if not filename == "":
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

        plt.clf()
