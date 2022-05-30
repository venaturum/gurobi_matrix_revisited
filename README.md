![The Matrix: Revisited](header.png)

## A problem proposed by Gurobi

> *"Look how binary is the form, the nature of things. Ones and zeros. Choice and its absence."*
> 
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &mdash;&nbsp; *Agent Smith, The Matrix Resurrections*

This repository provides models for solving "The Matrix: Revisited" problem, introduced by the folks from Gurobi in the following video: [Holiday Tech Talk - Santa’s Bag of Interesting & Unusual Optimization Applications](https://www.gurobi.com/resource/holiday-tech-talk-santas-bag-of-interesting-unusual-optimization-applications).

In this problem we have a target matrix $B$ and a pool of matrices $K = \lbrace M_{1}, M_{2}, \ldots, M_{|K|} \rbrace$.  We wish to take a linear combination of the matrices in the pool, using only integer coefficients, to produce a matrix "closest" to the target.  The pool of matrices, as introduced by Gurobi, is restricted to binary values but the problem does generalise to matrices over real numbers.

The notion of "closest" is defined to be either the L1 or L2 norm, and there are models implementing both.

The models are implemented using [*gurobipy*](https://pypi.org/project/gurobipy/), in particular the matrix API (it is only fitting in this context)!

Two heuristics are provided for producing an initial solution, both based on LP rounding, but implemented via linear regression with [*scikit-learn*](https://scikit-learn.org).

A comparison of results over a set of randomly generated problem instances can be found in */matrix_revisited/notebooks/analysis.ipynb*

## Modelling definitions

* $B$ : the target matrix of size $m \times n$
* $K = \lbrace M_{1}, M_{2}, \ldots, M_{|K|} \rbrace$: the set of matrices (of size $m \times n$) in the pool
* $I = \lbrace 1, 2, \ldots, mn \rbrace$
* $b =$ vec$(B)$, where *vec* is the [vectorizaton](https://en.wikipedia.org/wiki/Vectorization_(mathematics)) transformation
* $A = \begin{bmatrix}\vert & \vert & & \vert\\\ \text{vec}(M_{1}) & \text{vec}(M_{2}) & \cdots & \text{vec}(M_{|K|})\\\ \vert & \vert & & \vert \end{bmatrix}$


## Models using the L2 Norm
### MR_Quad

This is a Quadratic Program in which the objective is to minimise the L2 Norm of a vector of slack variables.

$$
\begin{aligned} 
\text{min } & \sum_{i \in I} s_{i}^{2}\\
\text{s.t. } &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s_{i} = b_{i}, & \forall i \in I,\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}.&
\end{aligned}
$$

#### MR_Quad (matrix notation)

$$
\begin{aligned} 
\text{min } & s^{T}s\\
\text{s.t. } &&\\
& Ax + s = b,\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}.&
\end{aligned}
$$


### MR_Cons_Relax_L2

This model is conceptually the same model as above, but implemented using Gurobi's [Model.feasRelaxS](https://www.gurobi.com/documentation/9.5/refman/py_model_feasrelaxs.html) method to minimise the sum of squares of constraint violations.  Note that $s$ variables, and therefore the objective functions, are not required for the implementation.


## Models using the L1 Norm

### MR_Cons_Relax_L1

This model is also implemented using Gurobi's `Model.feasRelaxS` method.  A simple parameter change permits the objective function be the total magnitude of constraint violations.  Again, the $s$ variables, and therefore the objective functions, are not required for the implementation.

$$
\begin{aligned} 
\text{min } & \sum_{i \in I} |s_{i}|&\\
\text{s.t. } &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s_{i} = b_{i}, & \forall i \in I,\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}.&
\end{aligned}
$$

#### MR_Cons_Relax_L1 (matrix notation)
$$
\begin{aligned} 
\text{min } & \lVert s \rVert_{1}&\\
\text{s.t. } &&\\
& Ax + s = b, & \\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}.&
\end{aligned}
$$


### MR_MILP_L1

This model introduces a second set of slack variables $s^{\prime}$, and associated constraints, to linearise the L1 Norm of $s$.

$$
\begin{aligned} 
\text{min } & \sum_{i \in I} s_{i}^{\prime}&\\
\text{s.t. } &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s_{i} = b, & \forall i \in I,\\
& s_{i}^{\prime} \geq s_{i}, & \forall i \in I,\\
& s_{i}^{\prime} \geq -s_{i}, & \forall i \in I,\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}, s^{\prime} \in \mathbb{R}_{\geq 0}^{|I|}.&
\end{aligned}
$$

#### MR_MILP_L1 (matrix notation)

$$
\begin{aligned} 
\text{min } & \unicode{x1D7D9}^{T} s^{\prime}&\\
\text{s.t. } &&\\
& Ax + s = b,&\\
& s^{\prime} - s \geq 0,&\\
& s^{\prime} + s \geq 0,&\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}, s^{\prime} \in \mathbb{R}^{|I|}_{\geq 0}.&
\end{aligned}
$$


### MR_MILP_L1_SOS

Here we utilise two sets of slack variables, both non-negative, which capture the positive or negative violation of the main family of constraints respectively.  These two sets of slack variables are linked through [SOS (type 1)](https://www.gurobi.com/documentation/9.5/refman/py_sos.html) constraints.

$$
\begin{aligned} 
\text{min } & \sum_{i \in I} (s_{i}^{+} + s_{i}^{-})&\\
\text{s.t. } &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s_{i}^{+} - s_{i}^{-} = b_{i}, & \forall i \in I,\\
& SOS_{1}(s_{i}^{+}, s_{i}^{-}), & \forall i \in I,\\
& x \in \mathbb{Z^{|K|}},&\\
& s^{+}, s^{-} \in \mathbb{R}_{\geq 0}^{|I|}.&
\end{aligned}
$$

#### MR_MILP_L1_SOS (matrix notation)

$$
\begin{aligned} 
\text{min } & \unicode{x1D7D9}^T (s^{+} + s^{-})&\\
\text{s.t. } &&\\
& Ax + s^{+} - s^{-} = b,&\\
& SOS_{1}(s_{i}^{+}, s_{i}^{-}), & \forall i \in I,\\
& x \in \mathbb{Z}^{|K|},&\\
& s^{+}, s^{-} \in \mathbb{R}^{|I|}_{\geq 0}.&
\end{aligned}
$$

## Initial solution heuristics

Two heuristics, based on linear regression with `scikit-learn` are provided, and referred to as "rounding" and "iterative rounding".  The first of these will perform a linear regression on the dataset $(A, b)$ (no intercept).  The coefficients found when performing this regression will correspond to a solution to a linear relaxation of either of the two "L2 Norm models" implemented.  The advantage of a linear regression here is that it enables a concise and clean implementation (at the expense of importing scikit-learn into the environment).

The "rounding" heuristic will round the linear regression coefficients to their nearest integer value, while the "iterative_rounding" heuristic will loop, rounding whichever variable is closest to integral, before reformulating the regression and refitting.

## Installation

This project requires Python 3.8, or above, [Poetry](https://python-poetry.org/docs/#installation), and a valid Gurobi license.

With Poetry installed on your system, clone this repository and run

    poetry install

in a terminal from the root directory.  This will create a virtual environment in which all project dependencies will be installed.  In addition, the project itself will be loaded as an *editable install*, meaning that it will have the look and feel of a package called *matrix_revisited*.

The virtual environment can be activated with 

    poetry shell

and includes gurobipy, numpy, scipy, pandas, scikit-learn, matplotlib, seaborn and ipykernel.  The addition of ipykernel facilitates the use of Jupyter notebooks in IDEs such as VS Code (recommended) and PyCharm.  JupyterLab can also be used, assuming it is installed elsewhere as it is not part of the environment defined for this project.


## Problem generation

The `matrix_revisited.problem_data` module provides several ways for instance generation, namely methods `example`, `read_from_file`, `generate_instance` and an infinite generator `infinite_random_instances`.  Each of these will return a *(matrices, target)* tuple.

The simplest of these is `example()` which will return a small problem, which incidentally corresponds to the example used in Gurobi's tech talk.
Under the hood, this method is simply calling `read_from_file("example.prob")`, but this file read method can be used on any problem file which is correctly defined (see *The .prob file format* section below).

Greater flexibility is provided with the `generate_instance` method which has the following signature: 

    generate_instance(m, n, number_of_matrices, min_target=0, max_target=9, sparsity=0.3)

This method allows a user to generate a single instance, specifying the matrix dimensions (m,n), the number of matrices in the pool, the minimum and maximum values in the target matrix, and the expected fraction of 1s in each of the 0-1 matrices in the pool (equivalent to the mean of the values).

Lastly, the `infinite_random_instances` can be used to continually generate random instances.  It can be used like so:

    >> from matrix_revisited.problem_data import infinite_random_instances
    >> matrices, target = next(infinite_random_instances)

Note that the random instances returned will be vectors, or rather matrices of size $m \times 1$, and can be thought of as the result of matrix vectorisations (except in the case of $m$ prime, which would not correspond to any matrix).

## Using models

Each of the models is defined by a class which wraps a `gurobipy.Model`.  These classes are derived from a common base class to leverage polymorphism.  They belong to the module `matrix_revisited.models` and share the same constructor signature.  The docstring for the signature can be examined with Python's `help` function, e.g.

    >> from matrix_revisited.models import MR_Quad
    >> help(MR_Quad.__init__)

Note that additional keyword arguments provided to the constructor will be assumed to correspond to parameters on the underlying Gurobi model, and set with `gurobipy.Model.setParam`.

Creating an object will cause the underlying model to be built - the `optimize()` method can be called.  Eg

    >> from matrix_revisited.problem_data import example
    >> matrices, target = example()
    >> mr_quad = MR_Quad(matrices, target)
    >> results = mr_quad.optimize()

Alternatively a class method, `run()`, is defined which will facilitate creating a model and optimising in one line, e.g.  

    >> results = MR_Quad.run(matrices, target)

The return from these functions is a dictionary, detailing a few key pieces of data.  The `run` method can also be used to execute the model repeatedly and aggregate the result, recording mean and standard deviation where relevant (e.g. runtime):

    >> results = MR_Quad.run(matrices, target, runs=10)

## Running experiments

Several methods are defined in `matrix_revisited.experiment` which provide various ways in which many models and heuristics can be combined, and run, multiple times to produce data.  They are each based on a method of instance generation from the `matrix_revisited.problem_data` module:

    run_random(models, init_heuristics, max_instances, timeout_mins, log_to_console, runs=1)

<p/>

    run_from_file(models, init_heuristics, filename, log_to_console, runs=1)

<p/>

    run_from_spec(models, init_heuristics, log_to_console, runs, m, n, number_of_matrices, min_target, max_target, sparsity)

These methods have in common the following parameters:

- `models`: a non-empty subset of classes (not objects!) from `matrix_revisited.models`
- `init_heuristics`: a non-empty subset of \{"skip", "rounding", "iterative_rounding"\}
- `runs`: an integer indicating the number of times to run the same model/heuristic/problem combination in order to provide aggregated values
- `log_to_console`: a boolean indicating whether to enable Gurobi's solver output.

Note the `run_random` method also has `max_instances` and `timeout_mins` parameters - one of which needs to be supplied as a stopping criteria.  A timeout will not cause the execution to stop immediately, but will prevent any further random instances from being generated.

These methods will return a list of dictionaries - each dictionary corresponds to a single optimisation, with key-value pairs corresponding to the model name, heuristic used, solver runtime, value of first solution found at root node, the best objective value, in addition to data related to the problem instance.  This list can be passed into a `pandas.DataFrame` constructor for purposes of easier exploration.

## Running from command line

Limited command line interface (CLI) functionality is provided as an alternative to running experiments using the API.
Upon installation of the project using Poetry, the command `matrix_revisited` will be added to the virtual environment which serves an entry point for the CLI.

For example, running the following in a terminal

    matrix_revisited --models MR_Quad MR_Cons_Relax_L2 --heuristics skip iterative_rounding --problem-random 5 --runs 10 --pickle exp1.pkl

results in the execution corresponding to the cartestian product of

- the two models implementing a L2 Norm
- skipping initial heuristic, and using iterative rounding
- five random problems
- ten repetitions per model/heuristic/problem

The result is 200 (= 2 x 2 x 5 x 10) optimisations whose intermediate results are stored as a list of dictionaries.  This list is passed into a `pandas.DataFrame` and saved as pickle file *matrix_revisited/pickles/exp1.pkl*.

## The .prob file format

Single problem instances can be defined in, and read, from file.  The format is based on INI files and read using Python's [configparser](https://docs.python.org/3/library/configparser.html) module.

The file should begin with a [Problem] section, which contains two keys: "target" and "matrices".  The values corresponding to these keys will represent a 2D matrix, and 3D matrix, respectively and be composed of lists of lists.  See */matrix_revisited/problem_files/example.prob* for an example of this format.  Note that the structure of these values is very similar to the string representation of numpy arrays (which can come in handy for creating problem instances).
