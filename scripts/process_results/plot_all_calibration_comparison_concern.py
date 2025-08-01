import itertools
import os

import pandas as pd
from matplotlib import pyplot as plt

from abm4problemod.loader import load_history, get_period_change_steps
from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel


def main():
    # Calibration alternatives
    local_ods = {'bc': ATBCRModel, 'atbcr': ATBCRModel, 'fj': FJModel,
                 'ba': BiasedAssimilationModel, 'ab': ATBCRModel}
    algorithms = {'DE': 'DE', 'SHADE': 'LSHADE-100-True-False',
                  'LSHADE': 'LSHADE-6-True-True',
                  'PSO': 'PSO-1.49618-1.49618-0.7298'}

    concern_thresholds = [0.6, 0.75, 0.9]
    daily_factor = 450
    max_ev = 30000
    pop_size = 100
    topic = 'mig'
    mc = 20

    os.makedirs('results/summary_calibration_results', exist_ok=True)
    history = load_history(f'data/{topic}/{topic}_history.csv')

    for concern_threshold, local_od in itertools.product(concern_thresholds,
                                                         local_ods.keys()):
        xlabels = pd.read_csv('data/simulation_periods.csv')['Subperiod']
        plt.xticks(range(0, len(history)), xlabels, rotation=45)
        plt.plot(history, marker='.', label='History')
        plt.ylabel(f'Proportion of population concerned')
        plt.title('Proportion of population concerned evolution\n'
                  f'({local_od.upper()}-concern threshold = {concern_threshold})')

        for algorithm in algorithms.keys():
            measure_time = get_period_change_steps(daily_factor)

            folder_name = (
                f'results/{algorithms[algorithm]}_calibration_dynlocal/'
                f'{local_od}_{daily_factor}_{concern_threshold}_'
                f'{mc}_{max_ev}_{pop_size}')

            if os.path.exists(folder_name):
                results = pd.read_csv(
                    f'{folder_name}/best_solution_macro_output.csv')

                results = results.drop(columns='Seed').groupby(['Step']).agg(
                    ['mean', 'std', 'min', 'max']).reset_index()

                concern = \
                    pd.DataFrame(results).sort_values(by=['Step']).reset_index(
                        drop=True)['Concern']['mean'].to_list()

                concern = [concern[i] for i in measure_time]

                plt.plot(concern, marker='.',
                         label=f'Sim. {algorithm}-{local_od.upper()}')

        plt.gca().set_yticklabels(
            ['{:.2f}%'.format(x * 100) for x in plt.gca().get_yticks()])
        plt.legend()
        plt.savefig(
            f"results/summary_calibration_results/{local_od}_{concern_threshold}_comparison.png",
            bbox_inches="tight")
        plt.clf()


if __name__ == '__main__':
    main()
