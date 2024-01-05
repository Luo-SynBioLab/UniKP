import os
from UniKP.build_vocab import WordVocab
from UniKP.pretrain_trfm import TrfmSeq2seq
from UniKP.utils import Seq_to_vec, smiles_to_vec,device_picker,save_models

from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
import pickle
import math

script_path = os.path.dirname(os.path.realpath(__file__))



if __name__ == '__main__':
    device=device_picker()
    res = np.array(pd.read_excel(os.path.join(script_path,'..','datasets','kcat_km_samples.xlsx'), sheet_name='main')).T
    Smiles = res[1]
    Value = res[0]
    for i in range(len(Value)):
        Value[i] = math.log(Value[i], 10)
    print(max(Value), min(Value))
    feature_path=os.path.join(script_path,'..','retrained','Kcat_Km_features_910.pkl')

    with open(feature_path, 'rb') as f:
        feature=pickle.load(f)
    feature = np.array(feature)
    Label = np.array(Value)
    save_models(feature,Label,round=5,model_label='Kcat_Km', save_dir=os.path.dirname(feature_path))
