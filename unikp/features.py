"""Substrate and enzyme representations shared by every UniKP script.

`smiles_to_vec` and `Seq_to_vec` used to be copy-pasted into each of the
twelve training scripts, which is how they drifted apart (different CUDA
device indices, different truncation messages).  They live here once.
"""

import gc
import os
import pickle
import re

import numpy as np
import torch

from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from unikp.paths import SMILES_TRFM, SMILES_VOCAB, prott5_path
from utils import split

PAD_INDEX = 0
UNK_INDEX = 1
EOS_INDEX = 2
SOS_INDEX = 3
MASK_INDEX = 4

SEQ_LEN = 220
MAX_RESIDUES = 1000


def _load_vocab():
    """Unpickle vocab.pkl, which was written with WordVocab defined in __main__.

    The pickle carries a "__main__.WordVocab" reference, so it only loads
    when the running __main__ module happens to expose that name.  That used
    to be true by accident -- every script did `from build_vocab import
    WordVocab` at top level.  Now that the import lives here, alias it
    explicitly rather than relying on the caller's namespace.
    """
    import __main__
    if not hasattr(__main__, 'WordVocab'):
        __main__.WordVocab = WordVocab
    return WordVocab.load_vocab(SMILES_VOCAB)


def resolve_device():
    """Torch device to run ProtT5 on.

    Defaults to cuda:0 rather than the cuda:1 that most scripts hardcoded --
    `torch.cuda.is_available()` says nothing about a second GPU existing, so
    cuda:1 raised "invalid device ordinal" on single-GPU machines.  Override
    with UNIKP_DEVICE (e.g. "cuda:1", "cpu").
    """
    requested = os.environ.get('UNIKP_DEVICE')
    if requested:
        return torch.device(requested)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def smiles_to_vec(Smiles):
    """Encode SMILES strings with the pretrained SMILES Transformer.

    Returns an (n, 1024) array.
    """
    vocab = _load_vocab()

    def get_inputs(sm):
        sm = sm.split()
        if len(sm) > 218:
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, UNK_INDEX) for token in sm]
        ids = [SOS_INDEX] + ids + [EOS_INDEX]
        seg = [1] * len(ids)
        padding = [PAD_INDEX] * (SEQ_LEN - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load(SMILES_TRFM, map_location='cpu'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    return trfm.encode(torch.t(xid))


def Seq_to_vec(Sequence):
    """Encode protein sequences with ProtT5-XL-UniRef50.

    Returns an (n, 1024) array of residue-averaged embeddings.

    `Sequence` is not modified.  The previous version truncated sequences
    longer than 1000 residues *in place*, so the caller's own list was
    silently replaced with the truncated strings and any result table built
    from it recorded mangled sequences.
    """
    from transformers import T5EncoderModel, T5Tokenizer

    truncated = [
        s[:500] + s[-500:] if len(s) > MAX_RESIDUES else s
        for s in Sequence
    ]
    sequences_Example = [' '.join(s) for s in truncated]

    model_dir = prott5_path()
    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_dir)
    gc.collect()
    device = resolve_device()
    print('Using device:', device)
    model = model.to(device)
    model = model.eval()

    features = []
    for i, seq in enumerate(sequences_Example):
        print('For sequence ', str(i + 1))
        seq = [re.sub(r"[UZOB]", "X", seq)]
        ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)

    # Mean over residues.  This was a triple-nested Python loop over
    # (sequence, embedding dimension, residue) -- roughly an hour of pure
    # interpreter overhead on the 17k-sample kcat set.  Accumulating in
    # float64 reproduces the old numbers.
    return np.stack([f.astype(np.float64).mean(axis=0) for f in features])


def build_features(sequences, smiles):
    """Fused (n, 2048) substrate + enzyme representation."""
    smiles_input = smiles_to_vec(smiles)
    sequence_input = Seq_to_vec(sequences)
    return np.concatenate((smiles_input, sequence_input), axis=1)


def load_or_build_features(cache_path, sequences, smiles):
    """Return cached features, computing and caching them on a first run.

    Feature extraction needs a GPU and takes hours, so every script used to
    ship with the extraction commented out and a hardcoded read of a cache
    file that is not in the repository -- leaving no path from a fresh clone
    to a result.  This builds the cache when it is missing and reuses it
    afterwards.
    """
    if os.path.exists(cache_path):
        print('Loading cached features from', cache_path)
        with open(cache_path, 'rb') as f:
            return np.array(pickle.load(f))

    print('No feature cache at {}; extracting features now.'.format(cache_path))
    feature = build_features(sequences, smiles)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(feature, f)
    print('Wrote feature cache to', cache_path)
    return feature
