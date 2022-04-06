from abc import ABC, abstractmethod

import numpy as np
import sklearn
import pandas as pd


class Problem(ABC):

    @abstractmethod
    def sample_data(self, dist, s):
        ...


class NoisyFunction(Problem):

    def __init__(self, f, noise_dist):
        self.f = f
        self.noise_dist = noise_dist

    def sample_data(self, s: int, dist):
        x = dist(s)
        y = self.f(x) + self.noise_dist(s)
        return x, y


class HiddenParamFunction(Problem):
    def __init__(self, f):
        self.f = f

    def sample_data(self, s: int, dists):
        xs = [d(s) for d in dists]
        y = self.f(*xs)
        return xs[0], y


def gen_split_labels(x, n_split):
    """Generate split labels for easier visualisation."""
    if len(x) % n_split != 0:
        raise ValueError("n_split must be a multiple of the dataset length!")

    step = len(x) // n_split

    lb = [np.ones(step) * i for i in range(n_split)]
    return np.array(lb).flatten()


def gen_point_split_dataframe(x, y, n_split):
    lbs = gen_split_labels(x, n_split)
    return pd.DataFrame(np.stack([x, y, lbs]).T, columns=['x', 'y', 'num'])


def multi_model_fit(model, x, y, n_split: int):
    models = []

    if len(x) % n_split != 0:
        raise ValueError("n_split must be a multiple of the dataset length!")

    step = len(x) // n_split

    for i in range(n_split):
        m = sklearn.base.clone(model)
        xm = x[i * step:(i + 1) * step]
        ym = y[i * step:(i + 1) * step]
        m.fit(xm.reshape(-1, 1), ym)
        models.append(m)

    return models


def multi_model_eval(models, xs):
    yss = []
    for m in models:
        ys = m.predict(xs.reshape(-1, 1))
        yss.append(ys)
    return yss


def multi_model_eval_dataframe(xs, yss):
    yp = pd.DataFrame(np.array(yss).T, columns=range(len(yss))).melt(var_name='num',
                                                                     value_name='y')
    xp = pd.DataFrame(np.array(np.repeat(xs.reshape(1, -1), 10, 0)).T,
                      columns=range(10)).melt(var_name='num', value_name='x')
    comb = yp.reset_index().merge(xp.reset_index(), how='inner', on=['index', 'num'])
    return comb
