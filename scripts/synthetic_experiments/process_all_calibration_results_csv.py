import argparse
import os

import pandas as pd

from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel


def create_entry(folder_name, df_template, algorithm, local_od, target,
                 daily_factor, metric_columns, month_columns, constrained):
    solution = pd.read_csv(f'{folder_name}/front.csv').iloc[0]

    solution_df = df_template.copy()
    solution_df['c_value'] = [algorithm, local_od, target, daily_factor]
    solution_df['Mean'] = solution[metric_columns].values

    for i, month in enumerate(month_columns):
        if local_od == 'bc':
            solution_df[month] = [solution[f'convergence_month_{i}'],
                                  solution[f'threshold_bc_month_{i}'], '-',
                                  '-']
            solution_df['Params'] = ['Conv.', 'Bou.Thr.', '-', '-']
        elif local_od == 'atbcr':
            a = solution[f'threshold_bc_month_{i}']
            if constrained:
                b = solution[f'threshold_pol_month_{i}']
            else:
                b = solution[f'threshold_pol_month_{i}']
                b = a + (1 - a) * b
            solution_df[month] = [solution[f'convergence_month_{i}'], a, b,
                                  '-']
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
    parser = argparse.ArgumentParser(
        description="Run OD-ABM calibration to determine monthly local OD parameters:\n"
                    "Synthetic experiments with mean aggregation\n",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-u", "--unconstrained", action='store_true',
                        help="If set, calibration is unconstrained",
                        default=False)
    args = parser.parse_args()

    # Calibration alternatives
    local_ods = {'bc': ATBCRModel, 'atbcr': ATBCRModel, 'fj': FJModel,
                 'ab': ATBCRModel, 'ba': BiasedAssimilationModel}

    algorithm = 'DE'
    targets = ['stable', 'sin1', 'sin1noise']
    concern_thresholds = [0.6, 0.75, 0.9]

    daily_factor = 450
    max_ev = 30000
    pop_size = 100
    mc = 20

    for concern_threshold in concern_thresholds:

        results_dir = f'results/synth/summary_calibration_results/Concern{concern_threshold}/'
        os.makedirs(results_dir, exist_ok=True)

        if args.unconstrained:
            results_dir += 'uncons_'
        else:
            results_dir += 'cons_'

        # Solution template to fill
        metric_columns = ['mae', 'mse', 'mape', 'r2']
        month_columns = ['Month1', 'Month2', 'Month3', 'Month4', 'Month5',
                         'Month6', 'Month7', 'Month8', 'Month9', 'Month10',
                         'Month11', 'Month12', 'Month13', 'Month14', 'Month15']

        df_template = pd.DataFrame(
            columns=['Config', 'c_value', 'Metric', 'Mean',
                     'Params'] + month_columns)
        df_template['Config'] = ['Algorithm', 'Local OD', 'Series',
                                 'Days/Step']
        df_template['Metric'] = metric_columns
        df_template['Params'] = ['Conv.', 'Bou.Thr.', 'Pol.Thr.', 'Sus.']

        for target in targets:
            # Create a DataFrame to store the results
            concern_all_results = pd.DataFrame(columns=df_template.columns)

            degroot_folder = f'results/synth/{target}_history/Concern{concern_threshold}/degroot_{daily_factor}_{mc}'
            if os.path.exists(degroot_folder):
                solution_df = create_entry(degroot_folder, df_template, '-',
                                           'DeGroot', target, daily_factor,
                                           metric_columns, month_columns,
                                           False)

                concern_all_results = pd.concat(
                    [concern_all_results, solution_df])

            for local_od in local_ods.keys():

                folder_name = f'results/synth/{target}_history/Concern{concern_threshold}/'

                if args.unconstrained:
                    folder_name += 'uncons_'
                else:
                    folder_name += 'cons_'

                folder_name += f'{local_od}_{daily_factor}_{mc}_{max_ev}_{pop_size}'

                if os.path.exists(folder_name):
                    solution_df = create_entry(folder_name, df_template,
                                               algorithm, local_od, target,
                                               daily_factor, metric_columns,
                                               month_columns,
                                               not args.unconstrained)

                    # Stack solution_df rows into concern_all_results
                    concern_all_results = pd.concat(
                        [concern_all_results, solution_df])

            concern_all_results.reset_index(drop=True, inplace=True)
            concern_all_results.to_csv(
                f'{results_dir}best_solutions_{target}.csv', index=False)


if __name__ == '__main__':
    main()
