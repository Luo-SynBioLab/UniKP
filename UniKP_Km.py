import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from unikp.features import load_or_build_features
from unikp.paths import data_path, output_path
from unikp.seeding import set_global_seed

KM_DATASET = 'Km_test_11722.pkl'
FEATURE_CACHE = ('Km', 'Km_features_11722_PreKcat.pkl')


def Kcat_predict(Ifeature_ini, Label_ini, seed):
    Ifeature, Ifeature_test, Label, Label_test = train_test_split(Ifeature_ini, Label_ini, test_size=0.2,
                                                                  random_state=seed)
    model = ExtraTreesRegressor(random_state=seed)
    model.fit(Ifeature, Label)
    Pre_label = model.predict(Ifeature_test)
    Pcc = np.corrcoef(Label_test, Pre_label)[1][0]
    RMSE = np.sqrt(mean_squared_error(Label_test, Pre_label))
    MAE = mean_absolute_error(Label_test, Pre_label)
    r2 = r2_score(Label_test, Pre_label)
    print(r2, Pcc, RMSE, MAE)


if __name__ == '__main__':
    seed = set_global_seed()
    # Dataset Load
    with open(data_path(KM_DATASET), 'rb') as file:
        datasets = pickle.load(file)
    sequence = datasets['Sequence']
    smiles = datasets['smiles']
    Label = datasets['log10_KM']
    print(len(smiles), len(Label))
    feature = load_or_build_features(output_path(*FEATURE_CACHE), sequence, smiles)
    feature = np.array(feature)
    Label = np.array(Label)
    Kcat_predict(feature, Label, seed)
