# Gurobi's Matrix Revisited Problem

This repository provides models for solving Gurobi's "Matrix Revisited" problem.

https://www.gurobi.com/resource/holiday-tech-talk-santas-bag-of-interesting-unusual-optimization-applications


In this problem we have a target matrix $B$ and a pool of matrices $K = \{M_{1}, M_{2}, \ldots, M_{|K|}\}$.  We wish to take a linear combination of the matrices in the pool, using only integer coefficients, to produce a matrix "closest" to the target.

The notion of "closest" is defined to be either the L1 or L2 norm, and there are models implementing both.

The models are implemented using [*gurobipy*](https://pypi.org/project/gurobipy/), in particular the matrix API (it is only fitting in this context)!

Two heuristics are provided for producing an initial solution, both based on linear regression with [*scikit-learn*](https://scikit-learn.org).

A comparison of results over a set of randomly generated problem instances can be found in */matrix_revisited/notebooks/analysis.ipynb*

## Modelling definitions

* $B$ : the target matrix of size $m \times n$
* $K = \{M_{1}, M_{2}, \ldots, M_{|K|}\}$: the set of matrices (of size $m \times n$) in the pool
* $I = \{1, 2, \ldots, mn\}$
* $b =$ vec$(B)$, where *vec* is the [vectorizaton](https://en.wikipedia.org/wiki/Vectorization_(mathematics)) transformation
* $A = \begin{bmatrix} \vert & \vert & & \vert\\ vec(M_{1}) & vec(M_{2}) & \cdots & vec(M_{|K|})\\ \vert & \vert & & \vert \end{bmatrix}$


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


## Installation

This project requires Python 3.8, or above, [Poetry](https://python-poetry.org/docs/#installation), and a valid Gurobi license.

With Poetry installed on your system, clone this repository and run

    poetry install

in a terminal from the root directory.  This will create a virtual environment that you can activate with 

    poetry shell


## TODO

TODO:
   - Introduction
   - Document folder structure
   - Document how to import models
   - Document how to run experiment
   - Document .prob file formate


