import json
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

KCAT_DATASET = 'Kcat_combination_0918_wildtype_mutant.json'
# One row per record in the JSON (17010); the label filter below leaves 16838.
FEATURE_CACHE = ('PreKcat_new', 'features_17010_PreKcat.pkl')



def Kcat_predict(Ifeature, Label, weights):
    kf = KFold(n_splits=5, shuffle=True)
    All_pre_label = []
    All_real_label = []
    for train_index, test_index in kf.split(Ifeature, Label):
        Train_data, Train_label = Ifeature[train_index], Label[train_index]
        Test_data, Test_label = Ifeature[test_index], Label[test_index]
        model = ExtraTreesRegressor()
        # , sample_weight=weights[train_index]
        model.fit(Train_data, Train_label, sample_weight=weights[train_index])
        Pre_label = model.predict(Test_data)
        All_pre_label.extend(Pre_label)
        All_real_label.extend(Test_label)
    res = pd.DataFrame({'Value': All_real_label, 'Predict_Label': All_pre_label})
    res.to_excel(output_path('DMW', 'DMW_No_Normalize_2_LDS_Kcat_5_cv'+'.xlsx'))


if __name__ == '__main__':
    # Dataset Load
    with open(data_path(KCAT_DATASET), 'r') as file:
        datasets = json.load(file)
    # print(len(datasets))
    # datasets = datasets[:50]
    sequence = [data['Sequence'] for data in datasets]
    Smiles = [data['Smiles'] for data in datasets]
    Label = [float(data['Value']) for data in datasets]
    ECNumber = [data['ECNumber'] for data in datasets]
    Organism = [data['Organism'] for data in datasets]
    Substrate = [data['Substrate'] for data in datasets]
    Type = [data['Type'] for data in datasets]
    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    Label = np.array(Label)
    print(max(Label), min(Label))
    # Feature Extractor
    feature = load_or_build_features(output_path(*FEATURE_CACHE), sequence, Smiles)
    # Input dataset
    feature_new = []
    Label_new = []
    sequence_new = []
    Smiles_new = []
    ECNumber_new = []
    Organism_new = []
    Substrate_new = []
    Type_new = []
    for i in range(len(Label)):
        if -10000000000 < Label[i] and '.' not in Smiles[i]:
            feature_new.append(feature[i])
            Label_new.append(Label[i])
            sequence_new.append(sequence[i])
            Smiles_new.append(Smiles[i])
            ECNumber_new.append(ECNumber[i])
            Organism_new.append(Organism[i])
            Substrate_new.append(Substrate[i])
            Type_new.append(Type[i])
    print(len(Label_new), min(Label_new), max(Label_new))
    feature_new = np.array(feature_new)
    Label_new = np.array(Label_new)
    # sl_label = [Label_new[i] for i in range(len(Label_new))]
    # weights = Smooth_Label(sl_label)
    weights = np.ones([len(Label_new)], dtype=float)
    for i in range(len(weights)):
        if Label_new[i] > 4:
            weights[i] = 2
    # sum_weights = np.sum(weights)
    # for i in range(len(weights)):
    #     weights[i] /= sum_weights
    # Modelling
    Kcat_predict(feature_new, Label_new, weights)
