# a simple inference script adapted from the original of README.md
from joblib import parallel_backend
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

model_basename = {
    "Kcat": "UniKP_for_kcat.pkl",
    "Km": "UniKP_for_Km.pkl",
    "Kcat_Km": "UniKP_for_kcat_Km.pkl",
}


@click.command()
@click.option(
    "-s", "--sequence", prompt=True, required=True, help="Protein Squence input"
)
@click.option("-m", "--smiles", prompt=True, required=True, help="SMILES input")
@click.option(
    "-w",
    "--weight",
    prompt=True,
    type=click.Choice(["Km", "Kcat", "Kcat_Km", "All"]),
    required=True,
    help="Weight model, `Km`, `Kcat`, `Kcat_Km` or `All` for all.",
)
@click.option(
    "-p", "--prefix", default="default", help="Prefix label of this prediction"
)
@click.option("-o", "--output_dir", default=".", help="Output directory")
@click.option(
    "-d",
    "--device",
    help="Device to use. "
    "`cuda` for CUDA, `mps` for Apple Silicon MPS, `cpu` for pure CPU. "
    "If unset, device is picked automatically.",
)
@click.option("-j", "--nprocs", default=0, type=int, help="Number of processors")
def main(sequence, smiles, weight, prefix, output_dir, device, nprocs):
    sequences = [sequence]
    Smiles = [smiles]

    device = device_picker(device)
    if not nprocs:
        nprocs = os.cpu_count()

    smiles_vec = smiles_to_vec(Smiles, device=device)
    seq_vec = Seq_to_vec(sequences, device=device)

    fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)

    if weight and weight in model_basename:
        weights = [weight]
    else:
        weights = [w for w in model_basename.keys()]

    for weight in weights:
        output = os.path.join(
            output_dir, f"{prefix}_Kinetic_parameters_predicted_label_{weight}.xlsx"
        )
        run_inference(sequences, Smiles, fused_vector, weight, output, nprocs)


def run_inference(sequences, Smiles, fused_vector, weight, output, nprocs):
    if not (weight in model_basename):
        print(f"ERROR: Invalid weight input: {weight} from {model_basename.keys()}")
        return

    model_path = os.path.join(DEFAULT_UNIKP_WEIGHT, model_basename[weight])

    print(f"Loading model: {model_path}")
    model = pickle.load(open(model_path, "rb"))
    with parallel_backend("loky", n_jobs=nprocs):
        Pre_label = model.predict(fused_vector)
        Pre_label_pow = [math.pow(10, Pre_label[i]) for i in range(len(Pre_label))]
        print(len(Pre_label_pow))
        res = pd.DataFrame(
            {"sequences": sequences, "Smiles": Smiles, "Pre_label": Pre_label_pow}
        )
        os.makedirs(os.path.dirname(output), exist_ok=True)
        res.to_excel(output)


if __name__ == "__main__":
    main()
