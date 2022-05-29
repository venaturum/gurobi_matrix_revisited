import numpy as np
from sklearn.linear_model import LinearRegression

heuristic_choices = (
    "skip",
    "rounding",
    "iterative_rounding",
)


def set_initial_solution(how, x, A, b):
    """Sets values of variables using a rounding heuristic based on linear regression

    Parameters
    ----------
    how : {"skip", "rounding", "iterative_rounding"}
        Indicates which heurstic to use for initial solution
    x : gurobipy.MVar
        A gurobi matrix variable of size (n,)
    A : numpy.ndarray
        of size (m,n)
    b : numpy.ndarray
        of size (m,)
    """
    assert how in heuristic_choices
    if how == "rounding":
        set_initial_solution_rounding_heuristic(x, A, b)
    elif how == "iterative_rounding":
        set_initial_solution_iterative_rounding_heuristic(x, A, b)


def set_initial_solution_rounding_heuristic(x, A, b):
    """Sets values of variables using a rounding heuristic based on linear regression

    Parameters
    ----------
    x : gurobipy.MVar
        A gurobi matrix variable of size (n,)
    A : numpy.ndarray
        of size (m,n)
    b : numpy.ndarray
        of size (m,)
    """
    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, b)
    heur_sol = lr.coef_.round().astype(int)
    x.setAttr("Start", heur_sol)


def set_initial_solution_iterative_rounding_heuristic(x, A, b):
    """Sets values of variables using an iterative rounding heuristic based on linear regression

    Parameters
    ----------
    x : gurobipy.MVar
        A gurobi matrix variable of size (n,)
    A : numpy.ndarray
        of size (m,n)
    b : numpy.ndarray
        of size (m,)
    """
    indices = list(range(A.shape[1]))
    fixed_indices = []
    fixed_vals = []
    lr = LinearRegression(fit_intercept=False)
    while indices:
        lr.fit(A, b)
        rounded = np.round(lr.coef_).astype(int)
        ind_to_remove = abs(lr.coef_ - rounded).argmin()
        fixed_indices.append(indices.pop(ind_to_remove))
        fixed_vals.append(rounded[ind_to_remove])
        if indices:
            b = b - A[:, ind_to_remove] * rounded[ind_to_remove]
            A = np.delete(A, ind_to_remove, axis=1)
    x.setAttr("Start", np.array(fixed_vals)[np.argsort(fixed_indices)])
