"""
Model-specific params (Model: LightGBM)
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool


preprocess_params = []


train_params = [
    {"name": "n_estimators",
     "type": int,
     "default": 1000,
     "help": "Number of estimators."
    },
    {"name": "max_depth",
     "type": int,
     "default": -1,
     "help": "Max depth."
    },
    {"name": "num_leaves",
     "type": int,
     "default": 31,
     "help": "Number of leaves."
    },
]


infer_params = []
