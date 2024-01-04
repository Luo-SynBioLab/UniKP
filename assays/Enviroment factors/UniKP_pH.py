from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pickle
import math


def Kcat_predict(Ifeature, Label):
    kf = KFold(n_splits=5, shuffle=True)
    All_pre_label = []
    All_real_label = []
    for train_index, test_index in kf.split(Ifeature, Label):
        Train_data, Train_label = Ifeature[train_index], Label[train_index]
        Test_data, Test_label = Ifeature[test_index], Label[test_index]
        model = ExtraTreesRegressor()
        model.fit(Train_data, Train_label)
        Pre_label = model.predict(Test_data)
        All_pre_label.extend(Pre_label)
        All_real_label.extend(Test_label)
    res = pd.DataFrame({'Value': All_real_label, 'Predict_Label': All_pre_label})
    res.to_excel('pH/pH_Kcat_5_cv.xlsx')


if __name__ == '__main__':
    # Dataset Load
    database = np.array(pd.read_excel('pH/Generated_pH_unified_smiles_636.xlsx')).T
    sequence = database[1]
    smiles = database[3]
    pH = database[5].reshape([len(smiles), 1])
    Label = database[4]
    for i in range(len(Label)):
        Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))
    # Feature Extractor
    # smiles_input = smiles_to_vec(smiles)
    # sequence_input = Seq_to_vec(sequence)
    # print(sequence_input.shape, sequence_input.shape, pH.shape)
    # feature = np.concatenate((smiles_input, sequence_input, pH), axis=1)
    # with open("pH/features_636_pH_PreKcat.pkl", "wb") as f:
    #     pickle.dump(feature, f)
    with open("pH/features_636_pH_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    # Modelling
    Kcat_predict(feature, Label)
