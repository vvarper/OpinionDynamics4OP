import os

import pandas as pd
from matplotlib import pyplot as plt

from abm4problemod.model import ATBCRModel


def main():
    local_ods = {'atbcr': ATBCRModel}
    algorithms = {'DE': 'DE',
                  'SHADE': 'LSHADE-100-True-False',
                  'L-SHADE': 'LSHADE-6-True-True',
                  'PSO': 'PSO-1.49618-1.49618-0.7298'}
    daily_factor = 450
    concern_threshold = 0.9
    max_ev = 30000
    pop_size = 100
    mc = 20
    starting_generation = 0
    os.makedirs('results/summary_calibration_results', exist_ok=True)

    for local_od in local_ods.keys():
        for algorithm in algorithms.keys():

            folder_name = (
                f'results/{algorithms[algorithm]}_calibration_dynlocal/'
                f'{local_od}_{daily_factor}_{concern_threshold}_'
                f'{mc}_{max_ev}_{pop_size}')

            if os.path.exists(folder_name):
                print(folder_name)

                generations = len([name for name in os.listdir(folder_name)
                                   if os.path.isfile(
                        os.path.join(folder_name, name))
                                   and name.startswith('FUN.')])

                population_evolution = []
                num_evs_by_iteration = []
                num_evs = 0
                for generation in range(generations):
                    front = pd.read_csv(f'{folder_name}/'
                                        f'FUN.{generation}', header=None,
                                        delimiter=r"\s+")
                    front.columns = ['mae', 'mse', 'mape', 'r2']

                    if algorithm == 'L-SHADE':
                        num_evs += len(front)
                    else:
                        num_evs += pop_size

                    if generation >= starting_generation:
                        population_evolution.append(front['mape'].tolist())
                        num_evs_by_iteration.append(num_evs)

                best_solution_evolution = [min(i) for i in
                                           population_evolution]
                plt.plot(num_evs_by_iteration,
                         best_solution_evolution,
                         label=f'{algorithm}', marker='o', markersize=1.5)

        plt.xlabel(f'Number of evaluations')
        plt.ylabel('Best fitness value (MAPE)')
        plt.legend()

        plt.savefig(
            f'results/summary_calibration_results/'
            f'fitness_evolution_{local_od}_{concern_threshold}.pdf',
            bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    main()
