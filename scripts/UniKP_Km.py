import os
from UniKP.build_vocab import WordVocab
from UniKP.pretrain_trfm import TrfmSeq2seq
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import random
import pickle
from sklearn.model_selection import train_test_split

from UniKP.utils import Seq_to_vec, device_picker, smiles_to_vec


script_path = os.path.dirname(os.path.realpath(__file__))

def Kcat_predict(Ifeature_ini, Label_ini):
    Myseed = random.randint(0, 1000)
    print(Myseed)
    Ifeature, Ifeature_test, Label, Label_test = train_test_split(Ifeature_ini, Label_ini, test_size=0.2,
                                                                  random_state=Myseed)
    model = ExtraTreesRegressor()
    model.fit(Ifeature, Label)
    Pre_label = model.predict(Ifeature_test)
    Pcc = np.corrcoef(Label_test, Pre_label)[1][0]
    RMSE = np.sqrt(mean_squared_error(Label_test, Pre_label))
    MAE = mean_absolute_error(Label_test, Pre_label)
    r2 = r2_score(Label_test, Pre_label)
    print(r2, Pcc, RMSE, MAE)


if __name__ == '__main__':
    # Dataset Load
    device=device_picker()
    with open(os.path.join(script_path,'..','datasets','Km_test_11722.pkl'), 'rb') as file:
        datasets = pickle.load(file)
    # datasets = datasets
    # print(datasets)
    sequence = datasets['Sequence']
    smiles = datasets['smiles']
    Label = datasets['log10_KM']
    print(len(smiles), len(Label))

    feature_path=os.path.join(script_path,'..','retrained','Km_features_11722_PreKcat.pkl')
    
    if os.path.exists(feature_path):
        smiles_input = smiles_to_vec(smiles, device=device)
        sequence_input = Seq_to_vec(sequence, device=device)
        feature = np.concatenate((smiles_input, sequence_input), axis=1)
        os.makedirs(os.path.dirname(feature_path),exist_ok=True)
        with open(feature_path, "wb") as f:
            pickle.dump(feature, f)
    else:
        with open(feature_path, 'rb') as f:
            feature=pickle.load(f)
    
    feature = np.array(feature)
    Label = np.array(Label)
    Kcat_predict(feature, Label)
