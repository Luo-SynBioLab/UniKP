import os
from UniKP.build_vocab import WordVocab
from UniKP.pretrain_trfm import TrfmSeq2seq
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import pickle


from UniKP.utils import device_picker, save_models


script_path = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    # Dataset Load

    device = device_picker()
    with open(
        os.path.join(script_path, "..", "datasets", "Km_test_11722.pkl"), "rb"
    ) as file:
        datasets = pickle.load(file)

    Label = datasets["log10_KM"]

    feature_path = os.path.join(
        script_path, "..", "retrained", "Km_features_11722_PreKcat.pkl"
    )

    with open(feature_path, "rb") as f:
        feature = pickle.load(f)

    feature = np.array(feature)
    Label = np.array(Label)

    save_models(
        feature,
        Label,
        round=5,
        model_label="Km",
        save_dir=os.path.dirname(feature_path),
    )
