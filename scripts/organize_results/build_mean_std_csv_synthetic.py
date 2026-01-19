import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from abm4problemod.loader import get_period_change_steps, load_history
from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel


def compute_fitness(results: pd.DataFrame, history, mc, measure_time):
    is_valid = np.isfinite(history)

    fitness = np.zeros(mc)

    for i in range(mc):
        output = results.loc[results['Seed'] == i]['Concern'].to_numpy()[
            measure_time]

        fitness[i] = mean_absolute_percentage_error(history[is_valid],
                                                    output[is_valid])

    return fitness


def fill_dataframe(folder_name, history, mc, measure_time, df_mean, df_std,
                   target, od):
    if os.path.exists(folder_name):
        results = pd.read_csv(f'{folder_name}/best_solution_macro_output.csv')

        fitness = compute_fitness(results, history, mc, measure_time)

        # Add fitness.mean to dataframe
        df_mean.loc[df_mean['Series'] == target, od] = fitness.mean()
        df_std.loc[df_std['Series'] == target, od] = fitness.std()


def main():
    local_ods = {'bc': ATBCRModel, 'atbcr': ATBCRModel, 'fj': FJModel,
                 'ba': BiasedAssimilationModel, 'ab': ATBCRModel}
    targets = ['stable', 'sin1', 'sin1noise']

    concern_thresholds = [0.6, 0.75, 0.9]
    daily_factor = 450
    max_ev = 30000
    pop_size = 100
    folder = 'synth'
    mc = 20

    measure_time = get_period_change_steps(daily_factor, folder=folder)
    os.makedirs('results/summary_calibration_results', exist_ok=True)

    for concern_threshold in concern_thresholds:

        df_mean = pd.DataFrame(
            columns=['Series', 'DeGroot', 'fj', 'ba', 'bc', 'ab', 'atbcr'])
        df_mean['Series'] = targets

        df_std = pd.DataFrame(
            columns=['Series', 'DeGroot', 'fj', 'ba', 'bc', 'ab', 'atbcr'])
        df_std['Series'] = targets

        for target in targets:
            history = load_history(f'data/{folder}/{target}_history.csv')
            history[0] = 1 - concern_threshold

            degroot_folder_name = f'results/synth/{target}_history/Concern{concern_threshold}/degroot_{daily_factor}_{mc}'

            fill_dataframe(degroot_folder_name, history, mc, measure_time,
                           df_mean, df_std, target, 'DeGroot')

            for od in local_ods.keys():
                folder_name = (
                    f'results/synth/{target}_history/Concern{concern_threshold}/'
                    f'uncons_{od}_{daily_factor}_'
                    f'{mc}_{max_ev}_{pop_size}')

                fill_dataframe(folder_name, history, mc, measure_time, df_mean,
                               df_std, target, od)

        df_mean.to_csv(
            f'results/synth/summary_calibration_results/mean_fitness_{concern_threshold}.csv',
            index=False)
        df_std.to_csv(
            f'results/synth/summary_calibration_results/std_fitness_{concern_threshold}.csv',
            index=False)


if __name__ == '__main__':
    main()
