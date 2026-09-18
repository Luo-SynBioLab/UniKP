import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from unikp.features import load_or_build_features
from unikp.paths import data_path, output_path
from unikp.seeding import set_global_seed

PH_DATASET = 'Generated_pH_unified_smiles_636.xlsx'
# The cache holds the 2048-dim substrate + enzyme representation only; the
# pH column is appended at use time so that the same cache also feeds the
# two-layer model in UniKP_pH_s2.py, whose baseline expects 2048 features.
FEATURE_CACHE = ('pH', 'features_636_pH_PreKcat.pkl')


def Kcat_predict(Ifeature, Label, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    All_pre_label = []
    All_real_label = []
    for train_index, test_index in kf.split(Ifeature, Label):
        Train_data, Train_label = Ifeature[train_index], Label[train_index]
        Test_data, Test_label = Ifeature[test_index], Label[test_index]
        model = ExtraTreesRegressor(random_state=seed)
        model.fit(Train_data, Train_label)
        Pre_label = model.predict(Test_data)
        All_pre_label.extend(Pre_label)
        All_real_label.extend(Test_label)
    res = pd.DataFrame({'Value': All_real_label, 'Predict_Label': All_pre_label})
    res.to_excel(output_path('pH', 'pH_Kcat_5_cv.xlsx'))


if __name__ == '__main__':
    seed = set_global_seed()
    # Dataset Load
    database = np.array(pd.read_excel(data_path(PH_DATASET))).T
    sequence = database[1]
    smiles = database[3]
    pH = database[5].reshape([len(smiles), 1])
    Label = database[4]
    for i in range(len(Label)):
        Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))
    # Feature Extractor
    feature = load_or_build_features(output_path(*FEATURE_CACHE), sequence, smiles)
    feature = np.concatenate((feature, pH), axis=1)
    # Modelling
    Kcat_predict(feature, Label, seed)
