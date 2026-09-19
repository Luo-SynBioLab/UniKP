"""Reproducible randomness for the UniKP scripts.

Every result in this repository comes from a randomised procedure -- the
train/test split, the K-fold shuffle and the ExtraTrees fit itself -- and
none of them used to be seeded, so no run could be reproduced.  Scripts now
draw their seed from here and pass it to each of those three places.

Set UNIKP_SEED to change it; the default is 42.

    UNIKP_SEED=7 python UniKP_kcat.py

Scripts that repeat a procedure N times (e.g. UniKP_kcat.py's ten
train/test splits, or the five regressors UniKP_kcat_model.py pickles)
offset the seed per repeat via `repeat_seed`, so the repeats still differ
from one another while the set of them is reproducible.
"""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def get_seed():
    """Seed for this run: UNIKP_SEED if set, else DEFAULT_SEED."""
    raw = os.environ.get('UNIKP_SEED')
    if raw is None:
        return DEFAULT_SEED
    try:
        return int(raw)
    except ValueError:
        raise ValueError('UNIKP_SEED must be an integer, got {!r}'.format(raw))


def repeat_seed(seed, i):
    """Seed for repeat `i` of a procedure run several times under `seed`."""
    return seed + i


def set_global_seed(seed=None):
    """Seed the `random` and `numpy.random` global generators.

    Returns the seed, so callers can report and reuse it.  Pass it on to
    KFold / train_test_split / ExtraTreesRegressor as well -- seeding the
    globals alone does not make scikit-learn deterministic.
    """
    if seed is None:
        seed = get_seed()
    random.seed(seed)
    np.random.seed(seed)
    print('Random seed: {} (set UNIKP_SEED to change)'.format(seed))
    return seed
