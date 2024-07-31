# Code from https://github.com/gsbDBI/bandits/blob/master/bandits/policytree.py
# Adapts PolicyTree from R to python.

import numpy as np
import re
from warnings import warn

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, PackageNotInstalledError
from rpy2.situation import get_r_home

numpy2ri.activate()

try:
    pt = importr("policytree")
except PackageNotInstalledError:
    warn(
        f"""
    The R install linked to rpy2 (i.e., the one that lives in {get_r_home()} does not have policytree installed).
    You'll still be able to use any functionality that does not depend on this package.
    """
    )

__all__ = ["PolicyTreeCSC"]


class PolicyTreeCSC:

    def __init__(self, depth=2, min_node_size=1):
        self.depth = depth
        self.min_node_size = min_node_size

    def fit(self, xs, scores):
        self.policy = pt.policy_tree(
            X=np.array(xs),
            Gamma=-scores,  # Negative scores for CSC.
            depth=self.depth,
            min_node_size=self.min_node_size,
        )
        return self

    def predict(self, xs, type="action.id"):
        if type not in ["action.id", "node.id"]:
            raise ValueError(
                "PolicyTree.predict argument 'type' must be 'action.id' or 'node.id'."
            )
        wopt = pt.predict_policy_tree(self.policy, np.array(xs), type=type)
        wopt = np.array(wopt, dtype=int) - 1  # zero-indexing
        return wopt

    def get_str(self, actions=None, covariates=None):
        if actions is None:
            num_actions = int(self.policy.rx2("n.actions"))
            actions = range(num_actions)
        if covariates is None:
            num_covariates = int(self.policy.rx2("n.features"))
            covariates = range(num_covariates)

        tree_str = str(self.policy).split("\n")

        for i, line in enumerate(tree_str):

            is_action_line = len(re.findall("action:.*", line)) > 0
            is_covariate_line = len(re.findall("split_variable:.*", line)) > 0
            is_action_list_line = line.startswith("Actions: ")

            if is_action_line:
                pre, post = line.split(":")
                action_index = int(post.strip()) - 1
                post = actions[action_index]
                tree_str[i] = f"{pre}: {post}"

            elif is_covariate_line:
                pre, mid, post = line.split(":")
                covariate_index = int(mid.split(" ")[1][1:]) - 1
                covariate_name = covariates[covariate_index]
                split_value = post.strip()
                tree_str[i] = f"{pre}: {covariate_name} <= {split_value}"

            elif is_action_list_line:
                tree_str[i] = ""

        tree_str = "\n".join(tree_str)
        return tree_str
