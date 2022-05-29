from abc import ABC, abstractmethod

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from matrix_revisited import heuristic


def get_A_b_from_matrices_target(matrices, target):
    rows = matrices.shape[1] * matrices.shape[2]
    A = matrices.transpose((1, 2, 0)).reshape(rows, -1)
    b = target.reshape(rows)
    return A, b


class MR_Base(ABC):

    name = None

    def __init__(
        self,
        matrices,
        target,
        name=None,
        init_heuristic="skip",
        **kwargs,
    ):
        """Constructs a model, ready to be optimized

        Parameters
        ----------
        matrices : numpy.ndarray
            Should be of size (pool_cardinality, m,n).
        target : numpy.ndarray
            Should be of size (m,n).
        name : str, default None
            Name given to model.  If None, then class default for model will be used.
        init_heuristic : {"skip", "rounding", "iterative_rounding"}, default="skip"
            Indicates which custom heuristic (if any) to use for initial solution.
        """
        super().__init__()

        if name:
            self.name = name

        self.init_root_sol_obj = np.nan
        self.init_heuristic = init_heuristic
        self.A, self.b = get_A_b_from_matrices_target(matrices, target)

        self.m = gp.Model(self.name)
        self._build()

        for param, value in kwargs.items():
            self.m.setParam(param, value)

    def _add_x_variables(self):
        self.x = self.m.addMVar(
            shape=self.A.shape[1],
            lb=float("-inf"),
            vtype=GRB.INTEGER,
            name="x",
        )

    def _add_slack_variables(self):
        pass

    def _set_objective(self):
        pass

    @abstractmethod
    def _add_constraints(self):
        pass

    def _set_start_solution(self):
        heuristic.set_initial_solution(self.init_heuristic, self.x, self.A, self.b)

    def _build(self):
        self._add_x_variables()
        self._add_slack_variables()
        self.m.update()
        self._set_objective()
        self._add_constraints()
        self._set_start_solution()

    def optimize(self):
        """Calls the optimize method of gurobi model and returns dictionary of relevant results.

        Returns
        -------
        dict
            key/values for model name, intial heuristic used, runtime of optimization, value of first solution found, and best objective value
        """
        self.m.optimize(self._generate_root_sol_callback())
        return {
            "model": self.m.ModelName,
            "init_heur": self.init_heuristic,
            "runtime": self.m.runtime,
            "init_root_sol": self.init_root_sol_obj,
            "obj_val": self.m.ObjVal,
        }

    def _generate_root_sol_callback(self):
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                if model.cbGet(GRB.Callback.MIPSOL_SOLCNT) == 0:
                    self.init_root_sol_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        return callback

    @staticmethod
    def _compress_batch_results(results):
        static_results = ("model", "init_heur")
        dynamic_results = ("runtime", "init_root_sol", "obj_val")
        agg_result = {key: results[0][key] for key in static_results}
        agg_result.update(
            {
                f"{key}_mean": np.mean([result[key] for result in results])
                for key in dynamic_results
            }
        )
        agg_result.update(
            {
                f"{key}_std": np.std([result[key] for result in results])
                for key in dynamic_results
            }
        )
        return agg_result

    @classmethod
    def run(cls, *args, runs=1, **kwargs):
        """Constructs and optimizes model

        Returns
        -------
        dict
            Returns result of optimize()
        """
        if runs == 1:
            new_instance = cls(*args, **kwargs)
            return new_instance.optimize()
        return cls._compress_batch_results(
            [cls.run(*args, runs=1, **kwargs, Seed=i) for i in range(runs)]
        )
        # return {
        #     "model": results[0]["model"],
        #     "init_heur": results[0]["init_heur"],
        #     "runtime (s)": np.mean([r["runtime (s)"]]),
        #     "init_root_sol": self.init_root_sol_obj,
        #     "obj_val": self.m.ObjVal,
        # }

    def __repr__(self):
        return type(self).__name__

    def __str__(self):
        return type(self).__name__


class MR_Quad(MR_Base):

    name = "MR_quad"

    def _add_slack_variables(self):
        self.s = self.m.addMVar(
            shape=self.A.shape[0],
            lb=float("-inf"),
            vtype=GRB.CONTINUOUS,
            name="s",
        )

    def _set_objective(self):
        self.m.setObjective(self.s @ self.s, GRB.MINIMIZE)

    def _add_constraints(self):
        self.m.addConstr(
            self.A @ self.x + self.s == self.b,
            name="constraints",
        )


class MR_Cons_Relax_L1(MR_Base):

    name = "MR_cons_relax_l1"
    relaxobjtype = 0

    def _add_constraints(self):
        self.m.addConstr(
            self.A @ self.x == self.b,
            name="constraints",
        )

    def optimize(self):
        self.m.feasRelaxS(
            relaxobjtype=self.relaxobjtype,
            minrelax=False,
            vrelax=False,
            crelax=True,
        )
        return super().optimize()


class MR_Cons_Relax_L2(MR_Cons_Relax_L1):

    name = "MR_cons_relax_l2"
    relaxobjtype = 1


class MR_MILP_L1(MR_Base):

    name = "MR_milp_l1"

    def _add_slack_variables(self):
        self.s = self.m.addMVar(
            shape=self.A.shape[0],
            lb=float("-inf"),
            vtype=GRB.CONTINUOUS,
            name="s",
        )
        self.s_abs = self.m.addMVar(
            shape=self.A.shape[0],
            vtype=GRB.CONTINUOUS,
            name="s",
        )

    def _set_objective(self):
        self.m.setObjective(
            np.ones(self.A.shape[0]) @ self.s_abs,
            GRB.MINIMIZE,
        )

    def _add_constraints(self):
        self.m.addConstr(
            self.A @ self.x + self.s == self.b,
            name="constraints",
        )
        self.m.addConstr(
            np.eye(self.A.shape[0]) @ self.s_abs - self.s >= 0,
            name="s_abs_1",
        )
        self.m.addConstr(
            np.eye(self.A.shape[0]) @ self.s_abs + self.s >= 0,
            name="s_abs_2",
        )


class MR_MILP_L1_SOS(MR_Base):

    name = "MR_milp_l1_sos"

    def _add_slack_variables(self):
        self.s_pos = self.m.addMVar(
            shape=self.A.shape[0],
            vtype=GRB.CONTINUOUS,
            name="s_pos",
        )
        self.s_neg = self.m.addMVar(
            shape=self.A.shape[0],
            vtype=GRB.CONTINUOUS,
            name="s_neg",
        )

    def _set_objective(self):
        self.m.setObjective(
            np.ones(self.A.shape[0]) @ self.s_pos
            + np.ones(self.A.shape[0]) @ self.s_neg,
            GRB.MINIMIZE,
        )

    def _add_constraints(self):
        self.m.addConstr(
            self.A @ self.x + self.s_pos - self.s_neg == self.b,
            name="constraints",
        )
        for s_pos, s_neg in zip(self.s_pos, self.s_neg):
            self.m.addSOS(GRB.SOS_TYPE1, [s_pos, s_neg])


model_dict = {
    model.__name__: model
    for model in [
        MR_Quad,
        MR_Cons_Relax_L1,
        MR_Cons_Relax_L2,
        MR_MILP_L1,
        MR_MILP_L1_SOS,
    ]
}
