# The goal of this file is to generate postanalysis simulation data for the charitable giving experiment.
# That is generate simulation data using data from the adaptive experiment for charitable giving.
# See https://github.com/gsbDBI/toronto.

import pandas as pd
import random
import numpy as np
from itertools import product
from typing import Any, Dict, Sequence, Hashable
import mord
from .dataset_generator_base import DatasetGenerator

# Helper code----------------------------------------------------------------------------------------------
# Get useful constants from https://github.com/gsbDBI/toronto/blob/master/toronto/constants.py.
CONTEXTS = [
    "male",
    "white",
    "last_donation",
    "religious_spiritual",
    "rural",
    "political_leaning",
    "age",
    "views_right_bear_arms",
    "views_global_warming",
    "views_abortion",
    "views_immigration",
    "news_fox",
    "news_cnn",
    #'news_wp',
    "news_wsj",
    #'social_media'
]

ARM_NAMES = [
    "aipac",
    "blm",
    "clinton",
    #'colin',
    "green",
    "nra",
    "peta",
    "planned",
    #'salvation',
    "zuckerberg",
]

NUM_ARMS = len(ARM_NAMES)
NUM_OUTCOMES = 21
ARM_STR_TO_INT = dict(zip(ARM_NAMES, range(len(ARM_NAMES))))
ARM_INT_TO_STR = dict(zip(range(len(ARM_NAMES)), ARM_NAMES))

# Get helper functions from https://github.com/gsbDBI/bandits/blob/master/bandits/compute.py


def draw(ps, seed=None):
    """
    A much faster alternative to np.random.choice.
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(size=(len(ps), 1))
    ps_cumsum = ps.cumsum(1)
    assert np.all(np.isclose(ps_cumsum[:, -1], 1))
    return np.argmax(u < ps_cumsum, axis=1)


# Get helper functions from https://github.com/gsbDBI/toronto/blob/master/toronto/simulation.py.


def fit_mord(xs, ws, yobs, K, alpha):
    """
    Fits an ordinal regression model of y on x, x^2, w, x*w.
    User predict_ordinal_regression to compute prediction probabilities.
    """
    xws = context_arm_interactions(xs, ws, K)
    model = mord.LogisticAT(alpha=alpha, max_iter=int(1e6))
    model.fit(xws, yobs)
    return model


def predict_ordinal_regression(model, xs, num_arms, num_outcomes):
    """
    Given an ordinal regression model, computes the probability
    """
    n = len(xs)
    yhat = np.full((n, num_arms, num_outcomes), fill_value=np.nan)
    for w in range(num_arms):
        ws = np.full(n, fill_value=w)
        xws = context_arm_interactions(xs, ws, num_arms)
        yhat[:, w] = model.predict_proba(xws)
    return yhat


def draw_data(clf, xs, size, repeat_arms=None):
    n = len(xs)
    idx = np.random.randint(n, size=size)
    xs = xs[idx]  # sample_empirical_distribution(xs, size)
    probs = predict_ordinal_regression(
        clf, xs, num_arms=NUM_ARMS, num_outcomes=NUM_OUTCOMES
    )
    ys = np.column_stack([draw(probs[:, i, :]) for i in range(NUM_ARMS)]) - 10
    mus = probs @ np.arange(-10, 11)
    return xs, ys, mus


def context_arm_interactions(xs, ws, K):
    """
    Computes a matrix of contexts, contexts squared,
    one-hot encoded arms, and multiplicative interactions betweeen
    arms and (linear) contexts, i.e., [x, x^2, w, x*w].
    """
    n = len(xs)
    ws_onehot = np.zeros((n, K))
    ws_onehot[np.arange(n), ws] = 1.0
    ws_onehot = ws_onehot[:, 1:]
    xws = [xs, ws] + [x * w for x, w in product(xs.T, ws_onehot.T)] + [xs**2]
    return np.column_stack(xws)


# New class to generate charitable giving data.
# See: https://github.com/gsbDBI/toronto/blob/master/postanalysis-with-pooled-data/simulations.py


class CharitableGivingDatasetGenerator(DatasetGenerator):
    def __init__(self, simulation_algo="Random"):
        if simulation_algo == "Random":
            simulation_algo = random.choice(
                [
                    "mord5",
                    "mord10",
                    "mord20",
                    "mord40",
                    "mord50",
                    "mord80",
                    "mord100",
                    "mord160",
                    "mord320",
                    "mord500",
                    "mord640",
                    "mord1280",
                    "mord2560",
                ]
            )
        self.simulation_algo = simulation_algo
        self.alpha = float(simulation_algo[4:])

        # Load data from pilot 2
        df_pilot = pd.read_csv("Toronto-Charity-Adaptive-Data-For-Update-total.csv")
        df_pilot = df_pilot[df_pilot["ws"].isin(ARM_NAMES)]

        # Load data from main experiment
        df_main = pd.read_csv("Toronto-Charity-Adaptive-Data-For-Update-total_2.csv")
        df_main = df_main[df_main["ws"].isin(ARM_NAMES)]

        # Pool pilot 2 data and main experiment data
        df = pd.concat([df_pilot, df_main])

        xs_original = df[CONTEXTS].values
        ws_original = df["ws"].map(ARM_STR_TO_INT).astype(int).values
        yobs_original = df["yobs"].values
        idx = np.random.randint(len(xs_original), size=len(xs_original))
        self.clf = fit_mord(
            xs_original[idx], ws_original[idx], yobs_original[idx], NUM_ARMS, self.alpha
        )
        self.xs_original = xs_original

    def generate_data(self, size=1000):
        xs_generated, ys_generated, mus_generated = draw_data(
            self.clf, self.xs_original, size=size
        )
        return xs_generated, ys_generated / 10

    def generate_data_with_mus(self, size=1000):
        xs_generated, ys_generated, mus_generated = draw_data(
            self.clf, self.xs_original, size=size
        )
        return xs_generated, ys_generated / 10, mus_generated / 10

    @property
    def params(self) -> Dict[str, Any]:
        return {"family": "CharitableGiving" + self.simulation_algo}
