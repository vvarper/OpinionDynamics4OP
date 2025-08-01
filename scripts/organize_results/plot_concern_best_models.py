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
    algorithms = {'DE': 'DE', 'SHADE': 'LSHADE-100-True-False',
                  'L-SHADE': 'LSHADE-6-True-True',
                  'PSO': 'PSO-1.49618-1.49618-0.7298'}
    daily_factor = 450
    max_ev = 30000
    pop_size = 100
    topic = 'mig'
    mc = 20

    configs = {0.6: {'bc': 'DE', 'atbcr': 'L-SHADE', 'fj': 'DE', 'ab': 'PSO',
                     'ba': 'L-SHADE'},
               0.75: {'bc': 'DE', 'atbcr': 'L-SHADE', 'fj': 'DE', 'ab': 'DE',
                      'ba': 'L-SHADE'

                      },
               0.9: {'bc': 'DE', 'atbcr': 'DE', 'fj': 'DE', 'ab': 'L-SHADE',
                     'ba': 'L-SHADE'}}

    measure_time = get_period_change_steps(daily_factor)
    os.makedirs('results/summary_calibration_results', exist_ok=True)
    history = load_history(f'data/{topic}/{topic}_history.csv')

    for concern_threshold in configs.keys():
        xlabels = pd.read_csv('data/simulation_periods.csv')['Subperiod']
        plt.xticks(range(0, len(history)), xlabels, rotation=45)
        plt.plot(history, marker='.', label='History')
        plt.ylabel(f'Proportion of population concerned')
        palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Switch colors between DeGroot and ATBCR
        palette[1], palette[3] = palette[3], palette[1]

        degroot_folder_name = f'results/DeGroot/{daily_factor}_{concern_threshold}_{mc}'
        plot_concern_model(degroot_folder_name, 'DeGroot solution',
                           measure_time, palette[1])

        for j, local_od in enumerate(configs[concern_threshold]):
            algorithm = configs[concern_threshold][local_od]
            folder_name = (
                f'results/{algorithms[algorithm]}_calibration_dynlocal/'
                f'{local_od}_{daily_factor}_{concern_threshold}_'
                f'{mc}_{max_ev}_{pop_size}')

            if local_od == 'bc':
                local_od = 'dw'

            plot_concern_model(folder_name,
                               f'Best {local_od.upper()} solution',
                               measure_time, palette[j + 2])

        # Set y-axis limits
        plt.gca().set_yticklabels(
            ['{:.2f}%'.format(x * 100) for x in plt.gca().get_yticks()])

        plt.title(f"Concern threshold = {concern_threshold}")

        plt.legend()
        plt.savefig(
            f"results/summary_calibration_results/best_{concern_threshold}.pdf",
            bbox_inches="tight")
        plt.clf()


if __name__ == '__main__':
    main()
