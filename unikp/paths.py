"""Central definition of every path the UniKP scripts read or write.

All paths are resolved against the repository root, so a script works no
matter which directory it is launched from.  Override any default with an
environment variable when data or model weights live outside the checkout:

    UNIKP_DATA_DIR    input datasets              (default: <repo>/datasets)
    UNIKP_OUTPUT_DIR  results written by scripts  (default: <repo>/results)
    UNIKP_PROTT5_DIR  ProtT5-XL-UniRef50 weights  (default: <repo>/prot_t5_xl_uniref50)
    UNIKP_MODEL_DIR   trained UniKP regressors    (default: <repo>/UniKP)
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_dir(var, default_subdir):
    return os.path.abspath(os.environ.get(var, os.path.join(REPO_ROOT, default_subdir)))


DATA_DIR = _env_dir('UNIKP_DATA_DIR', 'datasets')
OUTPUT_DIR = _env_dir('UNIKP_OUTPUT_DIR', 'results')
PROTT5_DIR = _env_dir('UNIKP_PROTT5_DIR', 'prot_t5_xl_uniref50')
MODEL_DIR = _env_dir('UNIKP_MODEL_DIR', 'UniKP')

# SMILES Transformer assets shipped with this repository.
SMILES_VOCAB = os.path.join(REPO_ROOT, 'vocab.pkl')
SMILES_TRFM = os.path.join(REPO_ROOT, 'trfm_12_23000.pkl')


def ensure_dir(path):
    """Create `path` (a directory) if it does not exist yet, then return it."""
    os.makedirs(path, exist_ok=True)
    return path


def data_path(*parts):
    """Absolute path to an input dataset under DATA_DIR.

    Raises FileNotFoundError with an actionable message when the file is
    missing, rather than letting pandas/pickle fail further down.
    """
    path = os.path.join(DATA_DIR, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(
            '{} not found.\nExpected it under the dataset directory {}. '
            'Set UNIKP_DATA_DIR if your copy lives elsewhere.'.format(path, DATA_DIR))
    return path


def output_path(*parts):
    """Absolute path for a file to be written under OUTPUT_DIR.

    The parent directory is created on demand, so scripts no longer fail
    because e.g. `PreKcat_new/` does not exist.
    """
    path = os.path.join(OUTPUT_DIR, *parts)
    ensure_dir(os.path.dirname(path))
    return path


def require_output(*parts):
    """Absolute path to a file another script was supposed to have written."""
    path = os.path.join(OUTPUT_DIR, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(
            '{} not found.\nIt is produced by an earlier step -- run that script '
            'first (e.g. UniKP_kcat_model.py writes the PreKcat_new/*_model.pkl '
            'baselines).'.format(path))
    return path


def prott5_path():
    """Directory holding the ProtT5-XL-UniRef50 weights.

    Missing weights are reported with the download link instead of a
    transformers stack trace.
    """
    if not os.path.isdir(PROTT5_DIR):
        raise FileNotFoundError(
            'ProtT5-XL-UniRef50 not found at {}.\nDownload it from '
            'https://zenodo.org/records/4644188 and unpack it there, or point '
            'UNIKP_PROTT5_DIR at your copy.'.format(PROTT5_DIR))
    return PROTT5_DIR


def model_path(*parts):
    """Absolute path to a trained UniKP regressor under MODEL_DIR."""
    path = os.path.join(MODEL_DIR, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(
            '{} not found.\nDownload the trained UniKP models from '
            'https://huggingface.co/HanselYu/UniKP/tree/main into {}, or point '
            'UNIKP_MODEL_DIR at your copy.'.format(path, MODEL_DIR))
    return path
