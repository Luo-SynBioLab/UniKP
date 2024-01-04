import os
from UniKP.build_vocab import WordVocab
from UniKP.pretrain_trfm import TrfmSeq2seq
from UniKP.utils import Seq_to_vec, smiles_to_vec,device_picker
from joblib import parallel_backend

from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pickle
import math

script_path = os.path.dirname(os.path.realpath(__file__))


def Kcat_predict(Ifeature, Label):
    kf = KFold(n_splits=5, shuffle=True)
    All_pre_label = []
    All_real_label = []
    for train_index, test_index in kf.split(Ifeature, Label):
        Train_data, Train_label = Ifeature[train_index], Label[train_index]
        Test_data, Test_label = Ifeature[test_index], Label[test_index]
        model = ExtraTreesRegressor()
        print(f'Fitting at #{i} round...')
        with parallel_backend('loky', n_jobs=os.cpu_count()):
            model.fit(Train_data, Train_label)
            Pre_label = model.predict(Test_data)
            All_pre_label.extend(Pre_label)
            All_real_label.extend(Test_label)
    res = pd.DataFrame({'Value': All_real_label, 'Predict_Label': All_pre_label})
    res.to_excel('Kcat_Km_5_cv.xlsx')


if __name__ == '__main__':
    device=device_picker()
    res = np.array(pd.read_excel(os.path.join(script_path,'..','datasets','kcat_km_samples.xlsx'), sheet_name='main')).T
    Smiles = res[1]
    sequences = res[2]
    Value = res[0]
    for i in range(len(Value)):
        Value[i] = math.log(Value[i], 10)
    print(max(Value), min(Value))
    feature_path=os.path.join(script_path,'..','retrained','Kcat_Km_features_910.pkl')
    if not os.path.exists(feature_path):
        smiles_input = smiles_to_vec(Smiles, device=device)
        sequence_input = Seq_to_vec(sequences, device=device)
        feature = np.concatenate((smiles_input, sequence_input), axis=1)
        os.makedirs(os.path.dirname(feature_path),exist_ok=True)
        with open(feature_path, "wb") as f:
            pickle.dump(feature, f)
    else:
        with open(feature_path, 'rb') as f:
            feature=pickle.load(f)
    feature = np.array(feature)
    Label = np.array(Value)
    Kcat_predict(feature, Label)
