import math

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from unikp.features import load_or_build_features
from unikp.paths import data_path, output_path
from unikp.seeding import set_global_seed

KCAT_KM_DATASET = 'kcat_km_samples.xlsx'
FEATURE_CACHE = ('Kcat_Km', 'Kcat_Km_features_910.pkl')


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
    res.to_excel(output_path('Kcat_Km', 'Kcat_Km_5_cv.xlsx'))


if __name__ == '__main__':
    seed = set_global_seed()
    res = np.array(pd.read_excel(data_path(KCAT_KM_DATASET), sheet_name='main')).T
    Smiles = res[1]
    sequences = res[2]
    Value = res[0]
    for i in range(len(Value)):
        Value[i] = math.log(Value[i], 10)
    print(max(Value), min(Value))
    feature = load_or_build_features(output_path(*FEATURE_CACHE), sequences, Smiles)
    feature = np.array(feature)
    Label = np.array(Value)
    Kcat_predict(feature, Label, seed)
