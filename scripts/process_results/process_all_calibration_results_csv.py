import itertools
import os

import pandas as pd

from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel


def create_entry(folder_name, df_template, algorithm, local_od,
                 concern_threshold, daily_factor, metric_columns,
                 month_columns):
    solution = pd.read_csv(f'{folder_name}/front.csv').iloc[0]

    solution_df = df_template.copy()
    solution_df['c_value'] = [algorithm, local_od, concern_threshold,
                              daily_factor]
    solution_df['Mean'] = solution[metric_columns].values

    for i, month in enumerate(month_columns):
        if local_od == 'bc':
            solution_df[month] = [solution[f'convergence_month_{i}'],
                                  solution[f'threshold_bc_month_{i}'], '-',
                                  '-']
            solution_df['Params'] = ['Conv.', 'Bou.Thr.', '-', '-']
        elif local_od == 'atbcr':
            solution_df[month] = [solution[f'convergence_month_{i}'],
                                  solution[f'threshold_bc_month_{i}'],
                                  solution[f'threshold_pol_month_{i}'], '-']
            solution_df['Params'] = ['Conv.', 'Bou.Thr.', 'Pol.Thr.', '-']
        elif local_od == 'ab':
            solution_df[month] = [solution[f'convergence_month_{i}'],
                                  solution[f'threshold_bc_month_{i}'],
                                  solution[f'gamma_month_{i}'], '-']
            solution_df['Params'] = ['Conv.', 'Bou.Thr.', 'Gamma', '-']
        elif local_od == 'ba':
            solution_df[month] = [solution[f'bias_month_{i}'], '-', '-', '-']
            solution_df['Params'] = ['Bias', '-', '-', '-']
        else:
            solution_df[month] = [solution[f'susceptibility_month_{i}'], '-',
                                  '-', '-']
            solution_df['Params'] = ['Sus.', '-', '-', '-']

    return solution_df


def main():
    # Calibration alternatives
    local_ods = {'bc': ATBCRModel, 'atbcr': ATBCRModel, 'fj': FJModel,
                 'ab': ATBCRModel, 'ba': BiasedAssimilationModel}
    algorithms = {'DE': 'DE', 'SHADE': 'LSHADE-100-True-False',
                  'LSHADE': 'LSHADE-6-True-True',
                  'PSO': 'PSO-1.49618-1.49618-0.7298'}
    daily_factor = 450
    concern_thresholds = [0.6, 0.75, 0.9]
    max_ev = 30000
    pop_size = 100
    mc = 20

    os.makedirs('results/summary_calibration_results', exist_ok=True)

    # Solution template to fill
    metric_columns = ['mae', 'mse', 'mape', 'r2']
    month_columns = ['Jan23', 'Feb23', 'Mar23', 'Apr23', 'May23', 'Jun23',
                     'Jul23', 'Aug23', 'Sep23', 'Oct23', 'Nov23', 'Dec23',
                     'Jan24', 'Feb24', 'Mar24']

    df_template = pd.DataFrame(columns=['Config', 'c_value', 'Metric', 'Mean',
                                        'Params'] + month_columns)
    df_template['Config'] = ['Algorithm', 'Local OD', 'Con.Thr.', 'Days/Step']
    df_template['Metric'] = metric_columns
    df_template['Params'] = ['Conv.', 'Bou.Thr.', 'Pol.Thr.', 'Sus.']

    for concern_threshold in concern_thresholds:
        # Create a DataFrame to store the results
        concern_all_results = pd.DataFrame(columns=df_template.columns)

        for local_od, algorithm in itertools.product(local_ods.keys(),
                                                     algorithms.keys()):

            folder_name = (
                f'results/{algorithms[algorithm]}_calibration_dynlocal/'
                f'{local_od}_{daily_factor}_{concern_threshold}_'
                f'{mc}_{max_ev}_{pop_size}')

            if os.path.exists(folder_name):
                solution_df = create_entry(folder_name, df_template,
                                           algorithm, local_od,
                                           concern_threshold, daily_factor,
                                           metric_columns, month_columns)

                # Stack solution_df rows into concern_all_results
                concern_all_results = pd.concat(
                    [concern_all_results, solution_df])

        degroot_folder = f'results/DeGroot/{daily_factor}_{concern_threshold}_{mc}'
        if os.path.exists(degroot_folder):
            solution_df = create_entry(degroot_folder, df_template, '-',
                                       'DeGroot', concern_threshold,
                                       daily_factor, metric_columns,
                                       month_columns)
            # Stack solution_df rows into concern_all_results
            concern_all_results = pd.concat(
                [concern_all_results, solution_df])

        concern_all_results.reset_index(drop=True, inplace=True)
        concern_all_results.to_csv(
            f'results/summary_calibration_results/best_solutions_{concern_threshold}.csv',
            index=False)


if __name__ == '__main__':
    main()
