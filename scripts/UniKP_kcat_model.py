import os
from UniKP.build_vocab import WordVocab
from UniKP.pretrain_trfm import TrfmSeq2seq
import json

import numpy as np
import pickle
import math
from UniKP.utils import Seq_to_vec, device_picker, smiles_to_vec, save_models


script_path = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':
    device=device_picker()
    
    with open(os.path.join(script_path,'..','datasets','Kcat_combination_0918_wildtype_mutant.json'), 'r') as file:
        datasets = json.load(file)
    # print(len(datasets))
    Label = [float(data['Value']) for data in datasets]
    Smiles = [data['Smiles']for data in datasets]
    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)

    # Feature Extractor
    feature_path=os.path.join(script_path,'..','retrained','features_16838_PreKcat.pkl')
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)

    Label = np.array(Label)
    Label_new = []
    feature_new = []
    for i in range(len(Label)):
        if -10000000000 < Label[i] and '.' not in Smiles[i]:
            Label_new.append(Label[i])
            feature_new.append(feature[i])
    print(len(Label_new))
    Label_new = np.array(Label_new)
    feature_new = np.array(feature_new)
    save_models(
        feature_new, 
        Label_new,
        model_label='kcat', 
        save_dir=os.path.dirname(feature_path))
