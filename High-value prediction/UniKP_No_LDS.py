from collections import Counter
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt

import json
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pickle
import math

from UniKP.utils import get_lds_kernel_window



def Smooth_Label(Label_new):
    labels = Label_new
    for i in range(len(labels)):
        labels[i] = labels[i] - min(labels)
    bin_index_per_label = [int(label*4) for label in labels]
    # print(bin_index_per_label)
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    print(emp_label_dist, len(emp_label_dist))
    plt.figure(figsize=(6,4))
    # plt.savefig('emp_label_dist.jpg')
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    print(lds_kernel_window)
    eff_label_dist = convolve1d(np.array(emp_label_dist[:-10]), weights=lds_kernel_window, mode='constant')
    # plt.figure(figsize=(6,4))
    print(eff_label_dist, emp_label_dist[-10:])
    eff_label_dist = eff_label_dist.tolist()
    eff_label_dist.extend(emp_label_dist[-10:])
    # print(eff_label_dist)
    plt.bar([i for i in range(len(eff_label_dist))], eff_label_dist, color='r')
    plt.bar([i for i in range(len(emp_label_dist))], emp_label_dist, color='b')
    plt.savefig('Square_52_label_dist_distribution.jpg')
    print(eff_label_dist, len(eff_label_dist))
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x)*np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    # print(weights)
    return weights


def Kcat_predict(Ifeature, Label, weights):
    for i in range(3):
        kf = KFold(n_splits=5, shuffle=True)
        All_pre_label = []
        All_real_label = []
        for train_index, test_index in kf.split(Ifeature, Label):
            Train_data, Train_label = Ifeature[train_index], Label[train_index]
            Test_data, Test_label = Ifeature[test_index], Label[test_index]
            model = ExtraTreesRegressor()
            # , sample_weight=weights[train_index]
            model.fit(Train_data, Train_label)
            Pre_label = model.predict(Test_data)
            All_pre_label.extend(Pre_label)
            All_real_label.extend(Test_label)
        res = pd.DataFrame({'Value': All_real_label, 'Predict_Label': All_pre_label})
        res.to_excel(str(i+1)+'_No_LDS_Kcat_5_cv'+'.xlsx')


if __name__ == '__main__':
    # Dataset Load
    with open('Kcat_combination_0918_wildtype_mutant.json', 'r') as file:
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
    # smiles_input = smiles_to_vec(Smiles)
    # sequence_input = Seq_to_vec(sequence)
    # feature = np.concatenate((smiles_input, sequence_input), axis=1)
    with open("PreKcat_new/features_17010_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
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
        if Label_new[i] > 5:
            weights[i] = 2
    sum_weights = np.sum(weights)
    for i in range(len(weights)):
        weights[i] /= sum_weights
    # Modelling
    Kcat_predict(feature_new, Label_new, weights)
