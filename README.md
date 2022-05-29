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

#### MR_Quad

$$
\begin{aligned} 
\text{min} & \sum_{i \in I} s_is_i&\\
\text{s.t.} &&\\
& \sum_{k \in K} a_{ik}x_{ik} + s_i = b_i, & \forall i \in I,\\
& x \in \mathbb{Z},&\\
& s \in \mathbb{R}.&\\
\end{aligned}
$$