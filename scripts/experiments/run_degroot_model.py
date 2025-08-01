# Process arguments from command line #########################################
import argparse
import json
import os

import networkx as nx
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

from abm4problemod.calibration.problem import ODABMCalibrationProblem
from abm4problemod.loader import load_history, get_period_change_steps, \
    load_opinions
from abm4problemod.model import FJModel


def check_positive(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive int value")
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive int value")
    return ivalue


def main():
    # Read arguments from command line ########################################
    parser = argparse.ArgumentParser(
        description="Run DeGroot model:\n",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("daily_steps", type=check_positive,
                        help="Daily time-steps", default=450)
    parser.add_argument("concern_threshold", type=float,
                        help="Concern threshold")
    parser.add_argument("-m", "--mc", type=check_positive,
                        help="Number of Monte Carlo runs", default=20)

    args = parser.parse_args()
    topic = 'mig'

    output_dir = f'results/DeGroot'

    output_dir += (f'/{args.daily_steps}_{args.concern_threshold}_{args.mc}/')
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Load history and social network data ####################################
    history = load_history(f'data/{topic}/{topic}_history.csv')
    graph = nx.read_gml(f'data/{topic}/social_networks/barabasi_3.gml')
    # Read nodes id as integers
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    num_agents = len(graph.nodes)

    # Define simulation time ##################################################
    measure_time = get_period_change_steps(args.daily_steps)
    simulation_steps = measure_time[-1]

    # Establish fixed parameters of the model #################################
    initial_opinions = load_opinions(f'data/{topic}/{topic}01.csv',
                                     args.concern_threshold)

    fixed_parameters = {'num_agents': num_agents,
                        'initial_op': initial_opinions,
                        'edge_list': tuple(graph.edges),
                        'concern_threshold': args.concern_threshold,
                        'simulation_steps': simulation_steps}

    metrics = {'mae': mae, 'mse': mse, 'mape': mape, 'r2': r2}
    model = FJModel

    # Create problem instance #################################################
    problem = ODABMCalibrationProblem(model, fixed_parameters, history,
                                      measure_time, metrics, args.mc, True,
                                      num_processes=args.mc,
                                      generator=np.random.default_rng(seed=0),
                                      polarization=False, algorithmic=False)

    # Evaluate DeGroot ########################################################

    solution = problem.create_solution_from_values(
        [1.0] * len(problem.calib_parameters))
    results = problem.evaluate(solution)

    # Build a dataframe with the solution, storing objectives and variables
    import pandas as pd
    results = [list(solution.variables) + list(solution.objectives)]

    df = pd.DataFrame(results, columns=problem.calib_parameters + list(
        problem.metrics.keys()))

    # Save to csv
    df.to_csv(f'{output_dir}/front.csv', index=False)

    best_solution = df.iloc[0]

    # Save calibration info to JSON file
    calibration_info = {'mc': args.mc, 'daily_steps': args.daily_steps,
                        'concern_threshold': args.concern_threshold,
                        'local_od': 'fj', 'topic': topic,
                        'MAE': best_solution['mae'].item(),
                        'MSE': best_solution['mse'].item(),
                        'MAPE': best_solution['mape'].item(),
                        'R2': best_solution['r2'].item(),
                        'variables': best_solution[
                            problem.calib_parameters].tolist()}

    with open(f'{output_dir}/calibration_info.json', 'w') as f:
        json.dump(calibration_info, f)


if __name__ == '__main__':
    main()
