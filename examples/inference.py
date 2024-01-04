# a simple inference script adapted from the original of README.md
import torch
from UniKP.build_vocab import WordVocab
from UniKP.pretrain_trfm import TrfmSeq2seq
from UniKP.utils import smiles_to_vec, Seq_to_vec, DEFAULT_UNIKP_WEIGHT, device_picker

import os
import numpy as np
import pandas as pd
import pickle
import math
import click


@click.command()
@click.option('-s','--sequence', help='Protein Squence input')
@click.option('-m','--smiles', help='SMILES input')
@click.option('-w','--weight', help='Weight model, `Km`, `Kcat`, or `Kcat_Km`')
@click.option('-d', '--device', help='Device to use.')
def main(sequence,smiles,weight,device):
    sequences = [sequence]
    Smiles = [smiles]

    device=device_picker()

    smiles_vec = smiles_to_vec(Smiles,device=device)
    seq_vec = Seq_to_vec(sequences,device=device)

    fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)

    model_basename={
        'Kcat': 'UniKP_for_kcat.pkl',
        'Km': 'UniKP_for_Km.pkl',
        'Kcat_Km': 'UniKP_for_kcat_Km.pkl'
        }
    
    assert weight in model_basename, f'Invalid weight input: {weight} from {model_basename.keys()}'

    model_path=os.path.join(DEFAULT_UNIKP_WEIGHT,model_basename[weight])
    print(f'Loading model: {model_path}')
    model = pickle.load(open(model_path, "rb"))
    
    Pre_label = model.predict(fused_vector)
    Pre_label_pow = [math.pow(10, Pre_label[i]) for i in range(len(Pre_label))]
    print(len(Pre_label_pow))
    res = pd.DataFrame({'sequences': sequences, 'Smiles': Smiles, 'Pre_label': Pre_label_pow})
    res.to_excel('Kinetic_parameters_predicted_label.xlsx')


if __name__ == '__main__':
    main()