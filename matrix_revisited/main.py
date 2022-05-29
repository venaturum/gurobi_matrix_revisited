import argparse
from pathlib import Path

import pandas as pd

from matrix_revisited import experiment, root_dir
from matrix_revisited.models import model_dict

problem_spec_params = [
    "m",
    "n",
    "number_of_matrices",
    "min_target",
    "max_target",
    "sparsity",
]
heuristic_choices = [
    "skip",
    "rounding",
    "iterative_rounding",
]

# matrix_revisited --models MR_MILP_L1 MR_Quad MR_Cons_Relax_L1 --heuristics skip rounding


def run(args):
    print(args.models)
    print(args.heuristics)
    print(args.problem_file)
    print(args.problem_spec)
    print(args.problem_random)
    print(args.timeout)
    print(args.log_to_console)


def run_method1():
    pass


def run_method2():
    pass


def _model(m):
    return model_dict[m]


def _validate_file(file):
    if file is None:
        return None
    assumed_file = root_dir / "problem_files" / file
    if Path(file).is_file():
        return file
    elif Path(assumed_file).is_file():
        return assumed_file
    raise OSError(f"Could not find file: {file} or {assumed_file}.")


def _make_spec_dict(spec):
    spec_params = dict(zip(problem_spec_params[:-1], map(int, spec[:-1])))
    spec_params["sparsity"] = float(spec[-1])
    return spec_params


def run(args):
    assert (
        sum(
            map(
                lambda x: x is not None,
                (args.problem_file, args.problem_spec, args.problem_random),
            )
        )
        == 1
    ), "Exactly one of --problem-spec, --problem-random or --problem-file, must be specified."

    if args.problem_file:
        results = experiment.run_from_file(
            args.models, args.heuristics, args.problem_file, args.log_to_console
        )
    elif args.problem_random:
        results = experiment.run_random(
            args.models,
            args.heuristics,
            args.problem_random,
            args.timeout,
            args.log_to_console,
        )
    elif args.problem_spec:
        specs = _make_spec_dict(args.problem_spec)
        results = experiment.run_from_spec(
            args.models, args.heuristics, args.log_to_console, **specs
        )
    if args.pickle:
        pd.DataFrame(results).to_pickle(root_dir / "pickles" / args.pickle)


def cli():
    parser = argparse.ArgumentParser(description="CLI Description")
    parser.add_argument(
        "--models",
        required=True,
        action="extend",
        nargs="+",
        type=_model,
        choices=model_dict.values(),
    )
    parser.add_argument(
        "--heuristics",
        required=True,
        action="extend",
        nargs="+",
        type=str,
        choices=heuristic_choices,
    )
    parser.add_argument("--problem-file", type=_validate_file, default=None)
    parser.add_argument(
        "--problem-spec",
        action="extend",
        nargs=len(problem_spec_params),
        type=float,
        default=None,
    )
    parser.add_argument(
        "--problem-random",
        type=int,
        default=None,
        help="number of random problems to generate",
    )
    parser.add_argument("--timeout", type=int, default=None, help="timeout in minutes")
    parser.add_argument("--log-to-console", action="store_true")
    parser.add_argument("--pickle", type=str, default=None, help="name of pickle file")
    # TODO timeout only valid for --problem-random
    args = parser.parse_args()

    run(args)


# cli
# - run specific models
# - log to console
# - choose heuristics
# - problem file
# - problem spec (m,n,number of matrices, min, max)
# - experiment (timeout, runs per model)
# - pickle file
