# Gurobi's Matrix Revisited Problem

This repository provides models for solving Gurobi's "Matrix Revisited" problem.

https://www.gurobi.com/resource/holiday-tech-talk-santas-bag-of-interesting-unusual-optimization-applications

CONTINUE INTRO


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


### Models using the L2 Norm
#### MR_Quad

$$
\begin{aligned} 
\text{min } & \sum_{i \in I} s_i^2\\
\text{s.t. } &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s_i = b_i, & \forall i \in I,\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}.&
\end{aligned}
$$

##### MR_Quad (matrix notation)

$$
\begin{aligned} 
\text{min } & s^Ts\\
\text{s.t. } &&\\
& Ax + s = b,\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}.&
\end{aligned}
$$


#### MR_Cons_Relax_L2

Conceptually the same model as above, but implemented using Gurobi's [Model.feasRelaxS](https://www.gurobi.com/documentation/9.5/refman/py_model_feasrelaxs.html) method to minimise the sum of squares of constraint violations.  Note that $s$ variables are omitted in the implementation.


### Models using the L1 Norm

#### MR_Cons_Relax_L1

Implemented using Gurobi's [Model.feasRelaxS](https://www.gurobi.com/documentation/9.5/refman/py_model_feasrelaxs.html) method to minimise the total magnitude of constraint violations.  Note that $s$ variables are omitted in the implementation.

$$
\begin{aligned} 
\text{min } & \sum_{i \in I} |s_i|&\\
\text{s.t. } &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s_i = b_i, & \forall i \in I,\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}.&
\end{aligned}
$$

##### MR_Cons_Relax_L1 (matrix notation)
$$
\begin{aligned} 
\text{min } & \lVert s \rVert_1&\\
\text{s.t. } &&\\
& Ax + s = b, & \\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}.&
\end{aligned}
$$


#### MR_MILP_L1

$$
\begin{aligned} 
\text{min } & \sum_{i \in I} s^\prime_i&\\
\text{s.t. } &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s_i = b, & \forall i \in I,\\
& s^\prime_{i} \geq s_{i}, & \forall i \in I,\\
& s^\prime_{i} \geq -s_{i}, & \forall i \in I,\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}, s^\prime \in \mathbb{R}^{|I|}_{\geq 0}.&
\end{aligned}
$$

##### MR_MILP_L1 (matrix notation)

$$
\begin{aligned} 
\text{min } & \unicode{x1D7D9}^T s^\prime&\\
\text{s.t. } &&\\
& Ax + s = b,&\\
& s^\prime - s \geq 0,&\\
& s^\prime + s \geq 0,&\\
& x \in \mathbb{Z}^{|K|}, s \in \mathbb{R}^{|I|}, s^\prime \in \mathbb{R}^{|I|}_{\geq 0}.&
\end{aligned}
$$


#### MR_MILP_L1_SOS

$$
\begin{aligned} 
\text{min } & \sum_{i \in I} (s^{+}_i + s^{-}_i)&\\
\text{s.t. } &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s^{+}_i - s^{-}_i = b_i, & \forall i \in I,\\
& SOS_1(s^+_i, s^-_i), & \forall i \in I,\\
& x \in \mathbb{Z^{|K|}},&\\
& s^{+}, s^{-} \in \mathbb{R}^{|I|}_{\geq 0}.&
\end{aligned}
$$

##### MR_MILP_L1_SOS (matrix notation)

$$
\begin{aligned} 
\text{min } & \unicode{x1D7D9}^T (s^+ + s^-)&\\
\text{s.t. } &&\\
& Ax + s^{+} - s^{-} = b,&\\
& SOS_1(s^+_i, s^-_i), & \forall i \in I,\\
& x \in \mathbb{Z}^{|K|},&\\
& s^{+}, s^{-} \in \mathbb{R}^{|I|}_{\geq 0}.&
\end{aligned}
$$