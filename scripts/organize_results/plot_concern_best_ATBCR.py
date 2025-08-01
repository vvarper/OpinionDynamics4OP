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
    algorithm = 'DE'
    daily_factor = 450
    local_od = 'atbcr'
    max_ev = 30000
    pop_size = 100
    topic = 'mig'
    mc = 20
    concern_threshold = 0.9

    measure_time = get_period_change_steps(daily_factor)
    os.makedirs('results/summary_calibration_results', exist_ok=True)
    history = load_history(f'data/{topic}/{topic}_history.csv')

    xlabels = pd.read_csv('data/simulation_periods.csv')['Subperiod']

    plt.figure(figsize=(12, 6))

    plt.xlabel('Simulation period', fontsize=11)
    plt.xticks(range(len(xlabels[:-1])), xlabels[:-1], rotation=30, ha='left')

    plt.plot(history, marker='.', label='History')
    plt.ylabel(f'Proportion of population concerned', fontsize=11)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    folder_name = (
        f'results/{algorithm}_calibration_dynlocal/'
        f'{local_od}_{daily_factor}_{concern_threshold}_'
        f'{mc}_{max_ev}_{pop_size}')

    solution = pd.read_csv(f'{folder_name}/front.csv').iloc[0]
    conv = []
    thr_bc = []
    thr_pol = []

    for i, month in enumerate(xlabels[:-1]):
        conv.append(solution[f'convergence_month_{i}'])
        thr_bc.append(solution[f'threshold_bc_month_{i}'])
        thr_pol.append(solution[f'threshold_pol_month_{i}'])

    plot_concern_model(folder_name,
                       f'Best {local_od.upper()} solution',
                       measure_time, palette[1])

    # Set y-axis limits
    plt.gca().set_yticklabels(
        ['{:.2f}%'.format(x * 100) for x in plt.gca().get_yticks()])

    # Put a vertical line at each x-tick
    for i in range(len(xlabels)):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

    # Color the vertical space between x-ticks 2 and 3, and nalso between 4 and 5 in light green
    polarization_months = [4, 5, 8, 9, 11, 14]
    confidence_months = [3, 6, 10]

    for i in polarization_months:
        plt.axvspan(xmin=i, xmax=i + 1, color='red', linestyle='--', alpha=0.1)

    for i in confidence_months:
        plt.axvspan(xmin=i, xmax=i + 1, color='green', linestyle='--',
                    alpha=0.1)

    plt.text(-3, 0.14,
             f'Convergence Speed\nConfidence Threshold \nPolarization Threshold  ',
             ha='left',
             color='black')

    for i, month in enumerate(xlabels[:-1]):
        plt.text(i + 0.5, 0.14,
                 f'{conv[i]:.2f}\n{thr_bc[i]:.2f}\n{thr_pol[i]:.2f}',
                 ha='center',
                 color='black')

    plt.legend()
    plt.savefig(
        f"results/summary_calibration_results/best_atbcr.pdf",
        bbox_inches="tight")
    plt.clf()


if __name__ == '__main__':
    main()
