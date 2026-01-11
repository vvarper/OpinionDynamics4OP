import argparse
import itertools
import json
import os
import time

import networkx as nx
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

from abm4problemod.calibration.problem import ODABMCalibrationProblem
from abm4problemod.loader import load_history, get_period_change_steps
from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel
from abm4problemod.runner import mc_run


def plot_opinions(folder_name, target, daily_factor, local_od, mc,
                  concern_threshold, constrained=False):
    print(f'Processing {folder_name}...')
    with open(f'{folder_name}/calibration_info.json') as json_file:
        result_info = json.load(json_file)

        # Load history and social network data ########################
        history = load_history(f'data/synth/{target}_history.csv',
                               variable='concerned')
        history[0] = 1 - concern_threshold
        graph = nx.read_gml(f'data/synth/social_networks/barabasi_3.gml')

        # Read nodes id as integers
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        num_agents = len(graph.nodes)

        # Define simulation time ######################################
        measure_time = get_period_change_steps(daily_factor, folder='synth')
        simulation_steps = measure_time[-1]

        # Establish fixed parameters of the model #####################
        fixed_parameters = {'num_agents': num_agents,
                            'edge_list': tuple(graph.edges),
                            'simulation_steps': simulation_steps,
                            'concern_threshold': concern_threshold}

        metrics = {'mae': mae, 'mse': mse, 'mape': mape, 'r2': r2}
        model = ATBCRModel if local_od in ['atbcr', 'bc',
                                           'ab'] else FJModel if local_od in [
            'fj', 'DeGroot'] else BiasedAssimilationModel

        # Create problem instance #####################################

        problem = ODABMCalibrationProblem(model, fixed_parameters, history,
                                          measure_time, metrics, mc,
                                          True,
                                          'Concern',
                                          synth=True,
                                          num_processes=mc,
                                          generator=np.random.default_rng(
                                              seed=17),
                                          polarization=(local_od == 'atbcr'),
                                          algorithmic=(local_od == 'ab'),
                                          constrained=constrained)

        solution = result_info['variables']
        parameters = problem.decode_variables(solution)

        title = (f'Proportion of population concerned with best '
                 f'\n{local_od.upper()} (Series {target})')
        parameters['collector_full'] = True

        t1 = time.time()
        results = mc_run(model_cls=model, parameters=parameters, mc=mc,
                         synth=True, number_processes=mc,
                         data_collection_period=1)
        t2 = time.time()

        print(f"Simulation time: {t2 - t1:.2f} seconds")
        output_file = f'{folder_name}/best_solution_macro_output.csv'
        results.raw_global_results.to_csv(output_file, index=False)

        results.plot_concern(base_folder="data/synth",
                             measure_time=measure_time, history=history,
                             title=title,
                             filename=f'{folder_name}/best_solution_concern.png')

        sim_config = {'model_name': model.__name__,
                      'concern_threshold': concern_threshold,
                      'num_agents': num_agents,
                      'mc': -1}

        if local_od in ['bc', 'atbcr', 'ab']:
            sim_config['threshold_bc'] = parameters['threshold_bc'][0][1]
            sim_config['threshold_pol'] = parameters['threshold_pol'][0][1]
            if local_od == 'ab':
                sim_config['gamma'] = parameters['gamma'][0][1]
            else:
                sim_config['gamma'] = 0.0
        elif local_od in ['fj', 'DeGroot']:
            sim_config['susceptibility'] = parameters['susceptibility'][0][1]
        else:
            sim_config['bias'] = parameters['bias'][0][1]

        if local_od == 'DeGroot':
            title = (
                f'Opinions evolution with DeGroot model\n(Series {target})')
        else:
            title = (f'Opinions evolution with best '
                     f'\n{local_od.upper()} (Series {target})')

        results.plot_opinions(base_folder="data/synth", sim_config=sim_config,
                              measure_time=measure_time,
                              title=title,
                              filename=f'{folder_name}/best_solution_opinions.png')


def main():
    parser = argparse.ArgumentParser(
        description="Run OD-ABM calibration to determine monthly local OD parameters:\n"
                    "Synthetic experiments with mean aggregation\n",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-u", "--unconstrained", action='store_true',
                        help="If set, calibration is unconstrained",
                        default=False)
    args = parser.parse_args()

    local_ods = {'bc': ATBCRModel, 'atbcr': ATBCRModel, 'fj': FJModel,
                 'ab': ATBCRModel, 'ba': BiasedAssimilationModel}

    daily_factor = 450
    concern_thresholds = [0.6, 0.75, 0.9]
    max_ev = 30000
    pop_size = 100
    mc = 20
    target_series = ['stable', 'sin1', 'sin1noise']

    for concern_threshold, target in itertools.product(
            concern_thresholds, target_series):

        degroot_folder = f'results/synth/{target}_history/Concern{concern_threshold}/degroot_{daily_factor}_{mc}'

        if os.path.exists(degroot_folder):
            plot_opinions(degroot_folder, target, daily_factor,
                          'DeGroot', mc, concern_threshold, False)

        for local_od in local_ods.keys():
            folder_name = f'results/synth/{target}_history/Concern{concern_threshold}'
            if args.unconstrained:
                folder_name += f'/uncons_{local_od}_{daily_factor}_{mc}_{max_ev}_{pop_size}'
            else:
                folder_name += f'/cons_{local_od}_{daily_factor}_{mc}_{max_ev}_{pop_size}'

            if os.path.exists(folder_name + '/calibration_info.json'):
                plot_opinions(folder_name, target, daily_factor,
                              local_od, mc,
                              concern_threshold,
                              constrained=not args.unconstrained)


if __name__ == '__main__':
    main()
