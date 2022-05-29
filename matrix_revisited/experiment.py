import pandas as pd

from matrix_revisited.problem_data import (
    generate_instance,
    infinite_random_instances,
    read_problem_file,
)


def run_model(
    model, init_heuristic, matrices, target, instance_num=None, log_to_console=False
):
    result = model.run(
        matrices,
        target,
        init_heuristic=init_heuristic,
        OutputFlag=int(log_to_console),
        runs=10,
    )
    if instance_num:
        result["instance"] = instance_num
    result.update(
        dict(zip(("num_matrices", "matrices_m", "matrices_n"), matrices.shape))
    )
    return result


def run_models(
    models, init_heuristics, matrices, target, instance_num=None, log_to_console=False
):
    results = []
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
                )
            )
    return results


def run_random(models, init_heuristics, max_instances, timeout_mins, log_to_console):
    results = []
    if timeout_mins is None:
        timeout_mins = float("inf")
    now = pd.Timestamp.now()
    instance_num = 0

    def time_running():
        return (pd.Timestamp.now() - now) / pd.Timedelta("1m")

    while time_running() < timeout_mins and instance_num != max_instances:
        matrices, target = next(infinite_random_instances)
        instance_num += 1
        results.extend(
            run_models(
                models, init_heuristics, matrices, target, instance_num, log_to_console
            )
        )
    return results


def run_from_file(models, init_heuristics, filename, log_to_console):
    matrices, target = read_problem_file(filename)
    return run_models(
        models, init_heuristics, matrices, target, log_to_console=log_to_console
    )


def run_from_spec(
    models,
    init_heuristics,
    log_to_console,
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
        models, init_heuristics, matrices, target, log_to_console=log_to_console
    )
