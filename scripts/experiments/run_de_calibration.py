# Process arguments from command line #########################################
import argparse
import json
import os
import random

import networkx as nx
import numpy as np
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.comparator import ObjectiveComparator
from jmetal.util.evaluator import SequentialEvaluator
from jmetal.util.generator import RandomGenerator
from jmetal.util.observer import BasicObserver, WriteFrontToFileObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

from abm4problemod.calibration.algorithm.generalizedgde3 import GeneralizedGDE3
from abm4problemod.calibration.operator.crossover import \
    GeneralizedDifferentialEvolutionCrossover
from abm4problemod.calibration.problem import ODABMCalibrationProblem
from abm4problemod.loader import load_history, get_period_change_steps, \
    load_opinions
from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel


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
        description="Run OD-ABM calibration to determine monthly local OD parameters:\n",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("daily_steps", type=check_positive,
                        help="Daily time-steps", default=450)
    parser.add_argument("concern_threshold", type=float,
                        help="Concern threshold")
    parser.add_argument("max_evaluations", type=check_positive,
                        help="Max Evaluations of the optimizer")
    parser.add_argument("pop_size", type=check_positive,
                        help="Population Size of the optimizer")
    parser.add_argument("-c", "--cr", type=float, default=0.5,
                        help="Crossover rate of the optimizer")
    parser.add_argument("-f", "--f", type=float, default=0.5,
                        help="Mutation factor of the optimizer")
    parser.add_argument("-k", "--k", type=float, default=0.5,
                        help="Scaling factor of the optimizer")
    parser.add_argument("-l", "--local_od", type=str,
                        choices=['atbcr', 'bc', 'fj', 'ba', 'ab'],
                        help="Local OD model", default="atbcr")
    parser.add_argument("-m", "--mc", type=check_positive,
                        help="Number of Monte Carlo runs", default=20)
    parser.add_argument("-s", "--calibration_seed", type=int,
                        help="Seed for calibration", default=17)
    parser.add_argument("-d", "--dynamic_local", action='store_true',
                        help="Calibrate local parameters monthly or not")

    args = parser.parse_args()
    topic = 'mig'

    output_dir = f'results/DE_calibration'
    if args.dynamic_local:
        output_dir += '_dynlocal'

    output_dir += (f'/{args.local_od}_{args.daily_steps}_'
                   f'{args.concern_threshold}_{args.mc}_{args.max_evaluations}_'
                   f'{args.pop_size}')
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
    model = ATBCRModel if args.local_od in ['atbcr', 'bc',
                                            'ab'] else FJModel if args.local_od == 'fj' else BiasedAssimilationModel

    # Create problem instance #################################################
    problem = ODABMCalibrationProblem(model, fixed_parameters,
                                      history,
                                      measure_time,
                                      metrics, args.mc, args.dynamic_local,
                                      num_processes=args.mc,
                                      generator=np.random.default_rng(
                                          seed=args.calibration_seed),
                                      polarization=(args.local_od == 'atbcr'),
                                      algorithmic=(args.local_od == 'ab'))

    # Create DE Optimizer #####################################################

    optimizer = GeneralizedGDE3(problem=problem, population_size=args.pop_size,
                                cr=args.cr, f=args.f, k=args.k,
                                termination_criterion=StoppingByEvaluations(
                                    args.max_evaluations),
                                population_generator=RandomGenerator(),
                                population_evaluator=SequentialEvaluator(),
                                dominance_comparator=ObjectiveComparator(
                                    objectiveId=2),
                                crossover_operator=GeneralizedDifferentialEvolutionCrossover,
                                selection_operator=DifferentialEvolutionSelection,
                                crossover_kwargs={'max_diff': 0.9,
                                                  'min_diff': 0.1,
                                                  'comparing_idx': problem.idx_paired_variables(),
                                                  'generator': np.random.default_rng(
                                                      seed=args.calibration_seed)})

    optimizer.observable.register(observer=BasicObserver(frequency=1))
    optimizer.observable.register(WriteFrontToFileObserver(output_dir))

    # Run calibration #########################################################
    random.seed(args.calibration_seed)
    optimizer.run()

    # Print results ###########################################################

    front = optimizer.result()

    print("Algorithm: " + optimizer.get_name())
    print("Problem: " + problem.name())
    print("Computing time: " + str(optimizer.total_computing_time))

    # Build a dataframe with all optimizer.front solutions, storing objectives and variables
    import pandas as pd
    results = []
    for solution in front:
        results.append(list(solution.variables) + list(solution.objectives))
    df = pd.DataFrame(results, columns=problem.calib_parameters + list(
        problem.metrics.keys()))
    df = df.sort_values(by='mape')
    # Save to csv
    df.to_csv(f'{output_dir}/front.csv', index=False)

    best_solution = df.iloc[0]

    # Save calibration info to JSON file
    calibration_info = {'mc': args.mc, 'daily_steps': args.daily_steps,
                        'concern_threshold': args.concern_threshold,
                        'local_od': args.local_od, 'topic': topic,
                        'calibration_seed': args.calibration_seed,
                        'calib_metric': list(problem.metrics.keys())[2],
                        'optimizer': optimizer.get_name(),
                        'population_size': args.pop_size,
                        'max_evaluations': args.max_evaluations,
                        'algorithm': 'GDE3',
                        'cr': args.cr, 'f': args.f, 'k': args.k,
                        'computing_time': optimizer.total_computing_time,
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
