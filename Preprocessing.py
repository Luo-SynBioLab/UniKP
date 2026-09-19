"""Metrics over the per-sample predictions written by UniKP_kcat.py.

Reads one of the `*_all_samples_metrics.xlsx` files and reports Pcc / RMSE /
MAE / r2 over several slices of it.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from unikp.paths import OUTPUT_DIR

DEFAULT_RESULTS = os.path.join(OUTPUT_DIR, 'PreKcat_new', '1_all_samples_metrics.xlsx')

TEST = 1
TRAIN = 0


def load_results(path):
    """Load a results table, addressing columns by name rather than position."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            '{} not found.\nRun UniKP_kcat.py first, or pass --results with the '
            'path to an existing metrics table.'.format(path))
    df = pd.read_excel(path)
    expected = {'sequence', 'smiles', 'Type', 'Label', 'Predict_Label', 'Training or test'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError('{} is missing expected columns: {}'.format(path, sorted(missing)))
    return df


def report(name, value, predicted):
    value = np.asarray(value, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    print('***{}***'.format(name))
    if len(value) < 2:
        print('Only {} sample(s); metrics undefined.'.format(len(value)))
        return
    print('n:', len(value),
          'Pcc:', pearsonr(value, predicted)[0],
          'RMSE:', np.sqrt(mean_squared_error(value, predicted)),
          'MAE:', mean_absolute_error(value, predicted),
          'r2:', r2_score(value, predicted))


def evaluate(df, mask, name):
    subset = df[mask]
    report(name, subset['Label'], subset['Predict_Label'])


def novelty_masks(df):
    """Masks for test samples whose enzyme and/or substrate is unseen in training.

    Enzymes and substrates are tracked in *separate* sets.  The previous
    version pooled sequences and SMILES into one list, so the two vocabularies
    could cross-match, and combined them with `or` -- which flags a sample as
    novel when only one of the two is unseen.  Both readings are offered here
    explicitly, because "new substrate and enzyme" is ambiguous and the two
    give materially different test-set sizes.
    """
    train = df['Training or test'] == TRAIN
    train_sequences = set(df.loc[train, 'sequence'])
    train_smiles = set(df.loc[train, 'smiles'])

    is_test = df['Training or test'] == TEST
    new_enzyme = ~df['sequence'].isin(train_sequences)
    new_substrate = ~df['smiles'].isin(train_smiles)
    return {
        'Test, unseen enzyme': is_test & new_enzyme,
        'Test, unseen substrate': is_test & new_substrate,
        'Test, unseen enzyme AND unseen substrate': is_test & new_enzyme & new_substrate,
        'Test, unseen enzyme OR unseen substrate': is_test & (new_enzyme | new_substrate),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default=DEFAULT_RESULTS,
                        help='metrics table written by UniKP_kcat.py (default: %(default)s)')
    args = parser.parse_args()

    df = load_results(args.results)
    print('Loaded {} rows from {}\n'.format(len(df), args.results))

    is_test = df['Training or test'] == TEST
    is_wildtype = df['Type'] == 'wildtype'

    evaluate(df, slice(None), 'The whole set')
    evaluate(df, is_test, 'Test set')
    evaluate(df, is_wildtype, 'The whole wildtype set')
    evaluate(df, is_wildtype & is_test, 'The test wildtype set')
    evaluate(df, ~is_wildtype, 'The whole mutant set')
    evaluate(df, ~is_wildtype & is_test, 'The test mutant set')
    for name, mask in novelty_masks(df).items():
        evaluate(df, mask, name)


if __name__ == '__main__':
    main()
