import ast
from configparser import ConfigParser
from pathlib import Path

import numpy as np

from matrix_revisited import root_dir
from matrix_revisited.utils import run


def read_problem_file(filename):
    config = ConfigParser()
    config.read(filename)
    target = np.array(ast.literal_eval(config["Problem"]["target"]))
    matrices = np.array(ast.literal_eval(config["Problem"]["matrices"]))
    return matrices, target


def example():
    return read_problem_file(root_dir / "problem_files" / "example.prob")


# def example():
#     target = np.array(
#         [
#             [7, 8, 2, 0, 3],
#             [0, 4, 9, 5, 4],
#             [5, 4, 7, 0, 0],
#             [4, 4, 4, 1, 6],
#             [5, 1, 4, 5, 6],
#         ]
#     )

#     matrices = np.array(
#         [
#             [
#                 [1, 1, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 1, 0],
#                 [0, 0, 1, 1, 1],
#                 [1, 1, 1, 0, 0],
#             ],
#             [
#                 [0, 0, 1, 0, 0],
#                 [1, 0, 0, 0, 0],
#                 [0, 1, 0, 0, 0],
#                 [0, 0, 0, 0, 1],
#                 [0, 0, 0, 1, 0],
#             ],
#             [
#                 [1, 0, 0, 0, 0],
#                 [1, 1, 1, 1, 1],
#                 [0, 1, 1, 0, 0],
#                 [0, 0, 0, 0, 1],
#                 [0, 0, 0, 0, 0],
#             ],
#             [
#                 [0, 0, 0, 1, 1],
#                 [0, 0, 0, 0, 1],
#                 [0, 1, 1, 0, 0],
#                 [1, 0, 0, 0, 0],
#                 [1, 1, 1, 1, 0],
#             ],
#             [
#                 [0, 1, 1, 1, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 1, 0, 0],
#                 [0, 0, 1, 1, 1],
#             ],
#             [
#                 [1, 1, 0, 0, 0],
#                 [0, 1, 1, 0, 0],
#                 [0, 0, 1, 1, 0],
#                 [0, 0, 0, 1, 1],
#                 [0, 0, 1, 0, 0],
#             ],
#         ]
#     )

#     return matrices, target


def generate_instance(
    m, n, number_of_matrices, min_target=0, max_target=9, sparsity=0.3
):
    """Creates a set of 0-1 "option" matrices which will be multiplied by integers
    to try and get close to a target integer matrix.

    Parameters
    ----------
    m : int
        height of matrices
    n : int
        width of matrices
    number_of_matrices : int
        the number of "option" matrices
    min_target : int, optional
        a lower bound on values in the target matrix, by default 0
    max_target : int, optional
        an upper bound on values in the target matrix, by default 9
    sparsity : float, optional
        expected number of 1s in the 0-1 "option" matrices, as a fraction, by default 0.3

    Returns
    -------
    tuple of size 2: (matrices, target)
        both are numpy.ndarray
    """
    assert isinstance(min_target, (int, np.integer))
    assert isinstance(max_target, (int, np.integer))
    assert 0 < sparsity < 1

    target = np.random.choice(range(min_target, max_target + 1), size=(m, n))
    matrices = np.random.choice(
        [0, 1], size=(number_of_matrices, m, n), p=[1 - sparsity, sparsity]
    )

    return matrices, target


@run
def infinite_random_instances():
    while True:
        min_target = np.random.choice([1, 10, 20, 30, 40])
        matrices, target = generate_instance(
            m=np.random.choice(np.linspace(100, 1000, 10).astype(int)),
            n=1,
            min_target=min_target,
            max_target=min_target
            + np.random.choice(np.linspace(10, 30, 5).astype(int)),
            number_of_matrices=np.random.choice(np.linspace(20, 40, 3).astype(int)),
        )
        yield matrices, target


# @run
# def infinite_random_instances():
#     while True:
#         min_target = np.random.choice([1, 10])
#         matrices, target = generate_instance(
#             m=np.random.choice(np.linspace(25, 100, 6).astype(int)),
#             n=1,
#             min_target=min_target,
#             max_target=min_target + 10,
#             number_of_matrices=np.random.choice(np.linspace(6, 10, 5).astype(int)),
#         )
#         yield matrices, target
