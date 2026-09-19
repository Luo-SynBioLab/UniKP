import json
import math
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from unikp.features import load_or_build_features
from unikp.paths import data_path, output_path
from unikp.seeding import repeat_seed, set_global_seed

KCAT_DATASET = 'Kcat_combination_0918_wildtype_mutant.json'
FEATURE_CACHE = ('PreKcat_new', 'features_17010_PreKcat.pkl')


def Kcat_predict(Ifeature, Label, seed):
    # Five regressors fit to the same data; they differ only by RNG, so each
    # gets its own seed rather than all five collapsing to one model.
    for i in range(5):
        model = ExtraTreesRegressor(random_state=repeat_seed(seed, i))
        model.fit(Ifeature, Label)
        with open(output_path('PreKcat_new', str(i)+'_model.pkl'), "wb") as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    seed = set_global_seed()
    with open(data_path(KCAT_DATASET), 'r') as file:
        datasets = json.load(file)
    # print(len(datasets))
    sequence = [data['Sequence'] for data in datasets]
    Label = [float(data['Value']) for data in datasets]
    Smiles = [data['Smiles']for data in datasets]
    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    feature = load_or_build_features(output_path(*FEATURE_CACHE), sequence, Smiles)
    Label = np.array(Label)
    Label_new = []
    feature_new = []
    for i in range(len(Label)):
        if -10000000000 < Label[i] and '.' not in Smiles[i]:
            Label_new.append(Label[i])
            feature_new.append(feature[i])
    print(len(Label_new))
    Label_new = np.array(Label_new)
    feature_new = np.array(feature_new)
    Kcat_predict(feature_new, Label_new, seed)
