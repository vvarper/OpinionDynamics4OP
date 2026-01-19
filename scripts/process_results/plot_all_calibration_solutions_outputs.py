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
from abm4problemod.loader import load_history, get_period_change_steps, \
    load_opinions
from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel
from abm4problemod.runner import mc_run


def plot_opinions(folder_name, topic, daily_factor, concern_threshold,
                  local_od, mc, algorithm):
    print(f'Processing {folder_name}...')
    with open(f'{folder_name}/calibration_info.json') as json_file:
        result_info = json.load(json_file)

        # Load history and social network data ########################
        history = load_history(f'data/{topic}/{topic}_history.csv')
        graph = nx.read_gml(f'data/{topic}/social_networks/barabasi_3.gml')
        # Read nodes id as integers
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        num_agents = len(graph.nodes)

        # Define simulation time ######################################
        measure_time = get_period_change_steps(daily_factor)
        simulation_steps = measure_time[-1]

        # Establish fixed parameters of the model #####################
        initial_opinions = load_opinions(f'data/{topic}/'
                                         f'{topic}01.csv', concern_threshold)

        fixed_parameters = {'num_agents': num_agents,
                            'initial_op': initial_opinions,
                            'edge_list': tuple(graph.edges),
                            'concern_threshold': concern_threshold,
                            'simulation_steps': simulation_steps}

        metrics = {'mae': mae, 'mse': mse, 'mape': mape, 'r2': r2}
        model = ATBCRModel if local_od in ['atbcr', 'bc',
                                           'ab'] else FJModel if local_od in [
            'fj', 'DeGroot'] else BiasedAssimilationModel

        # Create problem instance #####################################

        problem = ODABMCalibrationProblem(model, fixed_parameters, history,
                                          measure_time, metrics, mc, True,
                                          'Concern', False,
                                          generator=np.random.default_rng(
                                              seed=17),
                                          polarization=(local_od == 'atbcr'),
                                          algorithmic=(local_od == 'ab'),
                                          constrained=True)

        solution = result_info['variables']
        parameters = problem.decode_variables(solution)

        parameters['collector_full'] = True

        t1 = time.time()
        results = mc_run(model_cls=model, parameters=parameters, mc=mc,
                         number_processes=mc, data_collection_period=1)
        t2 = time.time()

        print(f"Simulation time: {t2 - t1:.2f} seconds")
        output_file = f'{folder_name}/best_solution_macro_output.csv'
        results.raw_global_results.to_csv(output_file, index=False)

        if local_od == 'DeGroot':
            title = (f'Proportion of population concerned '
                     f'evolution with DeGroot model\n(daily factor '
                     f'{daily_factor}, concern threshold '
                     f'{concern_threshold})')
        else:
            title = (f'Proportion of population concerned '
                     f'evolution with best model '
                     f'\n{local_od.upper()}-{algorithm} (daily factor '
                     f'{daily_factor}, concern threshold '
                     f'{concern_threshold})')

        results.plot_concern(base_folder="data/mig", measure_time=measure_time,
                             history=history,
                             title=title,
                             filename=f'{folder_name}/best_solution_'
                                      f'concern.png')

        sim_config = {'model_name': model.__name__,
                      'concern_threshold': concern_threshold,
                      'num_agents': num_agents, 'mc': -1}

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
            title = (f'Opinions evolution with DeGroot model\n(daily '
                     f'factor {daily_factor}, concern '
                     f'threshold {concern_threshold})')
        else:
            title = (f'Opinions evolution with best '
                     f'{local_od.upper()}-{algorithm} model\n(daily '
                     f'factor {daily_factor}, concern '
                     f'threshold {concern_threshold})')

        results.plot_opinions(base_folder="data/mig", sim_config=sim_config,
                              measure_time=measure_time,
                              title=title,
                              filename=f'{folder_name}/best_solution_'
                                       f'opinions.png')


def main():
    local_ods = {'bc': ATBCRModel, 'atbcr': ATBCRModel, 'fj': FJModel,
                 'ab': ATBCRModel, 'ba': BiasedAssimilationModel}
    algorithms = {'DE': 'DE', 'SHADE': 'LSHADE-100-True-False',
                  'LSHADE': 'LSHADE-6-True-True',
                  'PSO': 'PSO-1.49618-1.49618-0.7298'}
    daily_factors = [45, 225, 450]
    concern_thresholds = [0.9]
    max_ev = 30000
    pop_size = 100
    mc = 20
    topic = 'mig'

    for concern_threshold, daily_factor in itertools.product(
            concern_thresholds, daily_factors):
        degroot_folder = f'results/DeGroot/{daily_factor}_{concern_threshold}_{mc}'

        if os.path.exists(degroot_folder):
            plot_opinions(degroot_folder, topic, daily_factor,
                          concern_threshold, 'DeGroot', mc, None)

        for local_od, algorithm in itertools.product(
                local_ods.keys(), algorithms.keys()):
            folder_name = (
                f'results/{algorithms[algorithm]}_calibration_dynlocal/'
                f'{local_od}_{daily_factor}_{concern_threshold}_'
                f'{mc}_{max_ev}_{pop_size}')

            if os.path.exists(folder_name):
                plot_opinions(folder_name, topic, daily_factor,
                              concern_threshold, local_od, mc, algorithm)


if __name__ == '__main__':
    main()
