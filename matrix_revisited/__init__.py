from pathlib import Path

import gurobipy_exceptions as gp_exc

gp_exc.patch_model_methods(convert=True)


root_dir = Path(__file__).parent.absolute()
