import json
import math
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from unikp.features import load_or_build_features
from unikp.paths import data_path, output_path

KCAT_DATASET = 'Kcat_combination_0918_wildtype_mutant.json'
# The cache holds one row per record in the JSON (17010); the label filter
# below is applied afterwards, leaving 16838 modelling samples.
FEATURE_CACHE = ('PreKcat_new', 'features_17010_PreKcat.pkl')


def Kcat_predict(Ifeature, Label, sequence_new, Smiles_new, ECNumber_new, Organism_new, Substrate_new, Type_new):
    for i in range(10):
        # Generate training or test set index
        ALL_index = [j for j in range(len(Ifeature))]
        train_index = np.array(random.sample(ALL_index, int(len(ALL_index)*0.9)))
        train_lookup = set(train_index.tolist())
        Training_or_test = np.array([0 if j in train_lookup else 1 for j in ALL_index])
        Train_data, Train_label = Ifeature[train_index], Label[train_index]
        model = ExtraTreesRegressor()
        model.fit(Train_data, Train_label)
        Pre_all_label = model.predict(Ifeature)
        res = pd.DataFrame({'sequence': sequence_new, 'smiles': Smiles_new, 'ECNumber': ECNumber_new,
                            'Organism': Organism_new, 'Substrate': Substrate_new, 'Type': Type_new,
                            'Label': Label, 'Predict_Label': Pre_all_label, 'Training or test': Training_or_test})
        res.to_excel(output_path('PreKcat_new', str(i+1)+'_all_samples_metrics.xlsx'))


if __name__ == '__main__':
    # Dataset Load
    with open(data_path(KCAT_DATASET), 'r') as file:
        datasets = json.load(file)
    # print(len(datasets))
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
    print(max(Label), min(Label))
    # Feature Extractor
    feature = load_or_build_features(output_path(*FEATURE_CACHE), sequence, Smiles)
    Label = np.array(Label)
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
    Label_new = np.array(Label_new)
    feature_new = np.array(feature_new)
    # Modelling
    Kcat_predict(feature_new, Label_new, sequence_new, Smiles_new, ECNumber_new,
                Organism_new, Substrate_new, Type_new)
