import warnings
from datetime import datetime, timedelta

import gurobipy_exceptions as gp_exc

from matrix_revisited.problem_data import (
    generate_instance,
    infinite_random_instances,
    read_problem_file,
)


def run_model(
    model,
    init_heuristic,
    matrices,
    target,
    instance_num=None,
    log_to_console=False,
    runs=1,
):
    result = model.run(
        matrices,
        target,
        init_heuristic=init_heuristic,
        OutputFlag=int(log_to_console),
        runs=runs,
    )
    if instance_num:
        result["instance"] = instance_num
    result.update(
        dict(zip(("num_matrices", "matrices_m", "matrices_n"), matrices.shape))
    )
    return result


def run_models(
    models,
    init_heuristics,
    matrices,
    target,
    instance_num=None,
    log_to_console=False,
    runs=1,
):
    results = []
    try:
        for model in models:
            for init_heuristic in init_heuristics:
                results.append(
                    run_model(
                        model,
                        init_heuristic,
                        matrices,
                        target,
                        instance_num,
                        log_to_console,
                        runs,
                    )
                )
    except gp_exc.GRBSizeLimitExceeded:
        warnings.warn(
            f"Aborting instance {instance_num} for all models due to {model} reaching size limit.",
            RuntimeWarning,
        )
        results = []
    return results


def run_random(
    models, init_heuristics, max_instances, timeout_mins, log_to_console, runs=1
):
    results = []
    if timeout_mins is None:
        timeout_mins = float("inf")
    now = datetime.now()
    instance_num = 0

    def time_running():
        return (datetime.now() - now) / timedelta(minutes=1)

    while time_running() < timeout_mins and instance_num != max_instances:
        matrices, target = next(infinite_random_instances)
        instance_num += 1
        results.extend(
            run_models(
                models,
                init_heuristics,
                matrices,
                target,
                instance_num,
                log_to_console,
                runs,
            )
        )
    return results


def run_from_file(models, init_heuristics, filename, log_to_console, runs=1):
    matrices, target = read_problem_file(filename)
    return run_models(
        models,
        init_heuristics,
        matrices,
        target,
        log_to_console=log_to_console,
        runs=runs,
    )


def run_from_spec(
    models,
    init_heuristics,
    log_to_console,
    runs,
    m,
    n,
    number_of_matrices,
    min_target,
    max_target,
    sparsity,
):
    matrices, target = generate_instance(
        m, n, number_of_matrices, min_target, max_target, sparsity
    )
    return run_models(
        models,
        init_heuristics,
        matrices,
        target,
        log_to_console=log_to_console,
        runs=runs,
    )
