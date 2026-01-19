import os

import pandas as pd
from matplotlib import pyplot as plt

from abm4problemod.loader import load_history, get_period_change_steps


def plot_concern_model(folder_name, label, measure_time, color):
    if os.path.exists(folder_name):
        results = pd.read_csv(f'{folder_name}/best_solution_macro_output.csv')

        results = results.drop(columns='Seed').groupby(['Step']).agg(
            ['mean', 'std', 'min', 'max']).reset_index()

        results = pd.DataFrame(results).sort_values(by=['Step']).reset_index(
            drop=True)['Concern']

        mean_concern = results['mean'].to_list()
        std_concern = results['std'].to_list()

        mean_concern = [mean_concern[i] for i in measure_time]
        std_concern = [std_concern[i] for i in measure_time]

        plt.plot(mean_concern, marker='.', label=label, color=color)

        plt.fill_between(range(len(mean_concern)),
                         [mean_concern[i] - std_concern[i] for i in
                          range(len(measure_time))],
                         [mean_concern[i] + std_concern[i] for i in
                          range(len(measure_time))], alpha=0.2, color=color)


def main():
    # Calibration alternatives
    algorithms = {'DE': 'DE'}
    daily_factor = 450
    max_ev = 30000
    pop_size = 100
    folder = 'synth'
    mc = 20

    concern_thresholds = [0.6, 0.75, 0.9]
    targets = ['stable', 'sin1', 'sin1noise']
    local_ods = ['bc', 'atbcr', 'fj', 'ab', 'ba']

    measure_time = get_period_change_steps(daily_factor, folder=folder)
    os.makedirs('results/synth/summary_calibration_results', exist_ok=True)

    for target in targets:
        history_ori = load_history(f'data/{folder}/{target}_history.csv')
        for concern_threshold in concern_thresholds:
            history = history_ori.copy()
            history[0] = 1 - concern_threshold
            xlabels = pd.read_csv(f'data/{folder}/simulation_periods.csv')[
                'Subperiod']
            plt.xticks(range(0, len(history)), xlabels, rotation=45)
            plt.plot(history, marker='.', label='History')
            plt.ylabel(f'Proportion of population concerned')
            palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # Switch colors between DeGroot and ATBCR
            palette[1], palette[3] = palette[3], palette[1]

            degroot_folder_name = f'results/synth/{target}_history/Concern{concern_threshold}/degroot_{daily_factor}_{mc}'

            plot_concern_model(degroot_folder_name, 'DeGroot solution',
                               measure_time, palette[1])

            for j, local_od in enumerate(local_ods):
                folder_name = f'results/synth/{target}_history/Concern{concern_threshold}/uncons_{local_od}_{daily_factor}_{mc}_{max_ev}_{pop_size}'

                if os.path.exists(folder_name):

                    if local_od == 'bc':
                        local_od = 'dw'

                    plot_concern_model(folder_name,
                                       f'Best {local_od.upper()} solution',
                                       measure_time, palette[j + 2])

            # Set y-axis limits
            plt.gca().set_yticklabels(
                ['{:.2f}%'.format(x * 100) for x in plt.gca().get_yticks()])

            plt.title(f"Concern threshold = {concern_threshold}")

            if concern_threshold == 0.75:
                plt.ylabel('')
            if concern_threshold == 0.9:
                plt.legend()
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.savefig(
                f"results/synth/summary_calibration_results/{target}_best_{concern_threshold}.pdf",
                bbox_inches="tight")
            plt.clf()


if __name__ == '__main__':
    main()
