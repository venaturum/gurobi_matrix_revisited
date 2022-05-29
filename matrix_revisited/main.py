import argparse
from pathlib import Path

from matrix_revisited import experiment, root_dir
from matrix_revisited.heuristic import heuristic_choices
from matrix_revisited.models import model_dict
from matrix_revisited.problem_data import generate_instance_params


def _model(m):
    if m not in model_dict.keys():
        raise TypeError("f{m} is not valid for a --model parameter.")
    return m


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
    spec_params = dict(zip(generate_instance_params[:-1], map(int, spec[:-1])))
    spec_params["sparsity"] = float(spec[-1])
    return spec_params


def run(args):
    models = list(map(model_dict.get, args.models))
    if args.problem_file:
        results = experiment.run_from_file(
            models,
            args.heuristics,
            args.problem_file,
            args.log_to_console,
            args.runs,
        )
    elif args.problem_random:
        results = experiment.run_random(
            models,
            args.heuristics,
            args.problem_random,
            args.timeout,
            args.log_to_console,
            args.runs,
        )
    elif args.problem_spec:
        specs = _make_spec_dict(args.problem_spec)
        results = experiment.run_from_spec(
            models, args.heuristics, args.log_to_console, args.runs, **specs
        )
    if args.pickle:
        import pandas as pd

        pd.DataFrame(results).to_pickle(root_dir / "pickles" / args.pickle)


def cli():
    parser = argparse.ArgumentParser(description="CLI Description")
    parser.add_argument(
        "--models",
        required=True,
        action="extend",
        nargs="+",
        type=_model,
        choices=model_dict.keys(),
        help="Which models to run from.  Choose one or more from "
        + ", ".join(model_dict.keys()),
    )
    parser.add_argument(
        "--heuristics",
        required=True,
        action="extend",
        nargs="+",
        type=str,
        choices=heuristic_choices,
        help="Which heuristics to use for the initial solution.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--problem-file",
        type=_validate_file,
        default=None,
        help="Run the models on a problem specified in a file. See 'example.prob' for format.",
    )
    group.add_argument(
        "--problem-spec",
        action="extend",
        nargs=len(generate_instance_params),
        type=float,
        default=None,
        help="Run the models on a randomly generated problem instance specified by parameters {problem_spec_params}.",
    )
    group.add_argument(
        "--problem-random",
        type=int,
        default=None,
        help="Number of random problem instances to generate",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many times to run each model on the same instance (aggregating results).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in minutes.  Execution will continue until all models have run on the current problem instance.",
    )
    parser.add_argument(
        "--log-to-console",
        action="store_true",
        help="Whether to print Gurobi log to screen.",
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default=None,
        help="Name of pickle file for writing results.  Will be stored in 'pickles' directory.",
    )

    args = parser.parse_args()

    run(args)
