import itertools
import os

import pandas as pd
from matplotlib import pyplot as plt

from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel


def main():
    local_ods = {'bc': ATBCRModel, 'atbcr': ATBCRModel, 'fj': FJModel,
                 'ab': ATBCRModel, 'ba': BiasedAssimilationModel}
    algorithms = {'DE': 'DE',
                  'SHADE': 'LSHADE-100-True-False',
                  'LSHADE': 'LSHADE-6-True-True',
                  'PSO': 'PSO-1.49618-1.49618-0.7298'}
    daily_factor = 450
    concern_thresholds = [0.6, 0.75, 0.9]
    max_ev = 30000
    pop_size = 100
    mc = 20
    starting_generation = 10

    for concern_threshold, local_od, algorithm in itertools.product(
            concern_thresholds, local_ods.keys(), algorithms.keys()):
        folder_name = (
            f'results/{algorithms[algorithm]}_calibration_dynlocal/'
            f'{local_od}_{daily_factor}_{concern_threshold}_'
            f'{mc}_{max_ev}_{pop_size}')

        if os.path.exists(folder_name):
            print(folder_name)

            generations = len([name for name in os.listdir(folder_name)
                               if
                               os.path.isfile(os.path.join(folder_name, name))
                               and name.startswith('FUN.')])

            population_evolution = []
            num_sols_by_gen = []
            for generation in range(generations):
                front = pd.read_csv(f'{folder_name}/'
                                    f'FUN.{generation}', header=None,
                                    delimiter=r"\s+")
                front.columns = ['mae', 'mse', 'mape', 'r2']
                num_sols_by_gen.append(len(front))

                if generation >= starting_generation:
                    population_evolution.append(front['mape'].tolist())

            if algorithm != 'PSO':
                x_values = [[i] * num_sols_by_gen[i] for i in
                            range(starting_generation, generations)]

                x_values = [item for sublist in x_values for item in sublist]
                evolution_flatten = [item for sublist in population_evolution
                                     for item in sublist]

                plt.scatter(x_values, evolution_flatten, alpha=0.2, s=1,
                            label='Population')

            best_solution_evolution = [min(i) for i in population_evolution]
            plt.plot(range(starting_generation, generations),
                     best_solution_evolution, color='red',
                     label='Best Solution', marker='o', markersize=2)

            plt.xlabel(f'{algorithm} Iteration')
            plt.ylabel('Fitness value (MAPE)')
            plt.title(f'Fitness evolution of the {algorithm} calibration for '
                      f'{local_od.upper()} model \n(daily factor: '
                      f'{daily_factor}, concern threshold: '
                      f'{concern_threshold})')
            plt.legend()

            plt.savefig(f'{folder_name}/fitness_evolution.png',
                        bbox_inches='tight')
            plt.clf()


if __name__ == '__main__':
    main()
