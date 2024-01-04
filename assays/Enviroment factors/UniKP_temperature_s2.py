from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import random
import pickle
import math


def Kcat_predict(feature, pH, sequence, smiles, Label):
    # Generate index
    Train_Validation_index = random.sample(range(len(feature)), int(len(feature)*0.8))
    Test_index = []
    for i in range(len(feature)):
        if i not in Train_Validation_index:
            Test_index.append(i)
    Validation_index = random.sample(Train_Validation_index, int(len(Train_Validation_index)*0.2))
    Train_index = []
    for i in range(len(feature)):
        if i not in Validation_index and i not in Test_index:
            Train_index.append(i)
    print(len(Train_index), len(Validation_index), len(Test_index))
    Training_Validation_Test = []
    for i in range(len(feature)):
        if i in Train_index:
            Training_Validation_Test.append(0)
        elif i in Validation_index:
            Training_Validation_Test.append(1)
        else:
            Training_Validation_Test.append(2)
    Train_index = np.array(Train_index)
    Validation_index = np.array(Validation_index)
    Test_index = np.array(Test_index)
    print(Train_index.shape, Validation_index.shape, Test_index.shape)
    # First model
    print(feature[Train_index].shape, pH[Train_index].shape)
    model_1_input = np.concatenate((feature[Train_index], pH[Train_index]), axis=1)
    model_first = ExtraTreesRegressor()
    model_first.fit(model_1_input, Label[Train_index])
    # Second model
    with open("PreKcat_new/0_model.pkl", "rb") as f:
        model_base = pickle.load(f)
    Kcat_baseline = model_base.predict(feature[Validation_index]).reshape([len(Validation_index), 1])
    model_1_2_input = np.concatenate((feature[Validation_index], pH[Validation_index]), axis=1)
    Kcat_calibrated = model_first.predict(model_1_2_input).reshape([len(Validation_index), 1])
    kcat_fused = np.concatenate((Kcat_baseline, Kcat_calibrated), axis=1)
    model_second = LinearRegression()
    model_second.fit(kcat_fused, Label[Validation_index])
    # Final prediction
    model_1_3_input = np.concatenate((feature, pH), axis=1)
    Kcat_calibrated_3 = model_first.predict(model_1_3_input).reshape([len(feature), 1])
    Kcat_baseline_3 = model_base.predict(feature).reshape([len(feature), 1])
    kcat_fused_3 = np.concatenate((Kcat_baseline_3, Kcat_calibrated_3), axis=1)
    Predicted_value = model_second.predict(kcat_fused_3).reshape([len(feature)])
    Training_Validation_Test = np.array(Training_Validation_Test).reshape([len(feature)])
    pH = np.array(pH).reshape([len(Label)])
    Kcat_baseline_3 = np.array(Kcat_baseline_3).reshape([len(feature)])
    Kcat_calibrated_3 = np.array(Kcat_calibrated_3).reshape([len(feature)])
    print(Training_Validation_Test.shape)
    # save
    res = pd.DataFrame({'Value': Label,
                        'sequence': sequence,
                        'smiles': smiles,
                        'pH': pH,
                        'Prediction_first_base': Kcat_baseline_3,
                        'Prediction_first_pH': Kcat_calibrated_3,
                        'Prediction_second': Predicted_value,
                        'Training_Validation_Test': Training_Validation_Test})
    res.to_excel('degree/s2_degree_Kcat.xlsx')


if __name__ == '__main__':
    # Dataset Load
    database = np.array(pd.read_excel('degree/Generated_degree_unified_smiles_572.xlsx')).T
    sequence = database[1]
    smiles = database[3]
    pH = database[5]
    Label = database[4]
    for i in range(len(Label)):
        Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))
    pH = np.array(pH).reshape([len(Label), 1])
    with open("degree/features_572_degree_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    # Modelling
    Kcat_predict(feature, pH, sequence, smiles, Label)
