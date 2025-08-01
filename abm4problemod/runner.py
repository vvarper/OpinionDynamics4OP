import multiprocessing
from collections.abc import Mapping
from functools import partial
from multiprocessing import Pool
from typing import Any, Optional

from tqdm.auto import tqdm

from mesa.model import Model
import mesa

from abm4problemod.statistics import SimulationStatistics

multiprocessing.set_start_method("spawn", force=True)


def mc_run(
        model_cls: type[Model],
        parameters: Mapping[str, Any],
        initial_seed: int = 0,
        mc: int = 1,
        number_processes: Optional[int] = 1,
        data_collection_period: int = -1,
        display_progress: bool = True,
) -> SimulationStatistics:
    """MC run a mesa model under specific parameters with different random seeds.

    Parameters
    ----------
    model_cls : Type[Model]
        The model class to mc-run
    parameters : Mapping[str, Any],
        Dictionary with model parameters over which to run the model.
    number_processes : int, optional
        Number of processes used, by default 1. Set this to None if you want to use all CPUs.
    data_collection_period : int, optional
        Number of steps after which data gets collected, by default -1 (end of episode)
    display_progress : bool, optional
        Display batch run process, by default True

    Returns
    -------
    SimulationStatistics
    """

    seeds = range(initial_seed, initial_seed + mc)

    process_func = partial(
        _model_run_func,
        model_cls,
        parameters=parameters,
        data_collection_period=data_collection_period,
    )

    results: list[dict[str, Any]] = []

    with tqdm(total=len(seeds), disable=not display_progress) as pbar:
        if number_processes == 1:
            for run in seeds:
                data = process_func(run)
                results.extend(data)
                pbar.update()
        else:
            with Pool(number_processes) as p:
                for data in p.imap_unordered(process_func, seeds):
                    results.extend(data)
                    pbar.update()

    return SimulationStatistics(results)


def _model_run_func(
        model_cls: type[Model],
        seed: int,
        parameters: Mapping[str, Any],
        data_collection_period: int,
) -> list[dict[str, Any]]:
    """Run a single model run and collect model and agent data.

    Parameters
    ----------
    model_cls : Type[Model]
        The model class to batch-run
    seed: int
        The random seed for this run
    parameters : Mapping[str, Any]
        Dictionary with model parameters over which to run the model
    data_collection_period : int
        Number of steps after which data gets collected

    Returns
    -------
    List[Dict[str, Any]]
        Return model_data, agent_data from the reporters
    """
    model = model_cls(**parameters, seed=seed)
    while model.running:
        model.step()

    data = []

    steps = list(range(0, model._steps, data_collection_period))
    if not steps or steps[-1] != model._steps - 1:
        steps.append(model._steps - 1)

    for step in steps:
        model_data, all_agents_data = mesa.batchrunner._collect_data(model,
                                                                     step)

        # If there are agent_reporters, then create an entry for each agent
        if all_agents_data:
            stepdata = [
                {"Seed": seed, "Step": step, **model_data, **agent_data, } for
                agent_data in all_agents_data]
        # If there is only model data, then create a single entry for the step
        else:
            stepdata = [{"Seed": seed, "Step": step, **model_data, }]
        data.extend(stepdata)

    return data
