import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from unikp.features import load_or_build_features
from unikp.paths import data_path, output_path
from unikp.seeding import set_global_seed

KCAT_DATASET = 'Kcat_combination_0918_wildtype_mutant.json'
# One row per record in the JSON (17010); the label filter below leaves 16838.
FEATURE_CACHE = ('PreKcat_new', 'features_17010_PreKcat.pkl')



def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window


def Smooth_Label(Label_new):
    # Shift the labels so the smallest is zero, then bin them.
    #
    # This used to be an in-place loop that recomputed min(labels) on every
    # iteration, over the list it was modifying.  Once the loop reached the
    # original minimum that element became 0, the running minimum became 0
    # too, and every later element was shifted by nothing -- so only the
    # elements up to the original argmin were actually shifted, and the bins
    # (and therefore the re-weighting) were computed from a half-shifted
    # array.  Take the offset once, and do not alias the caller's list.
    offset = min(Label_new)
    labels = [label - offset for label in Label_new]
    bin_index_per_label = [int(label*4) for label in labels]
    # print(bin_index_per_label)
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    print(emp_label_dist, len(emp_label_dist))
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=3, sigma=1)
    print(lds_kernel_window)
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    print(eff_label_dist, emp_label_dist)
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    # print(weights)
    return weights


def Kcat_predict(Ifeature, Label, weights, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    All_pre_label = []
    All_real_label = []
    for train_index, test_index in kf.split(Ifeature, Label):
        Train_data, Train_label = Ifeature[train_index], Label[train_index]
        Test_data, Test_label = Ifeature[test_index], Label[test_index]
        model = ExtraTreesRegressor(random_state=seed)
        # , sample_weight=weights[train_index]
        model.fit(Train_data, Train_label, sample_weight=weights[train_index])
        Pre_label = model.predict(Test_data)
        All_pre_label.extend(Pre_label)
        All_real_label.extend(Test_label)
    res = pd.DataFrame({'Value': All_real_label, 'Predict_Label': All_pre_label})
    res.to_excel(output_path('LDS', '31_LDS_Kcat_5_cv'+'.xlsx'))


if __name__ == '__main__':
    seed = set_global_seed()
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
    # feature = feature[:50]
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
    sl_label = [Label_new[i] for i in range(len(Label_new))]
    weights = Smooth_Label(sl_label)
    # Modelling
    Kcat_predict(feature_new, Label_new, weights, seed)
