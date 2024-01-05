import os
import re
import gc
import torch
import math
import pickle
import torch.nn as nn
from rdkit import Chem
from rdkit import rdBase

import pooch
import zipfile


import numpy as np
from joblib import parallel_backend
from sklearn.ensemble import ExtraTreesRegressor

rdBase.DisableLog('rdApp.*')


dir_path = os.path.dirname(os.path.realpath(__file__))
# urls from weights
DEFAULT_PROT_T5_XL_UNI_REF50_WEIGHT_URL = 'doi:10.5281/zenodo.4644187/prot_t5_xl_uniref50.zip'
DEFAULT_PROT_T5_XL_UNI_REF50_WEIGHT_MD5 = 'md5:ab11a7eddfbaff5784effd41380b482a'

DEFAULT_UNIKP_WEIGTHS_URL={
    'https://huggingface.co/HanselYu/UniKP/resolve/main/UniKP%20for%20Km.pkl': 'md5:3e5e29dfabb0648448cb2fcd6f7cedd5',
    'https://huggingface.co/HanselYu/UniKP/resolve/main/UniKP%20for%20kcat.pkl':'md5:bf4e2c87deec0da8359ecb767e562bf2',
    'https://huggingface.co/HanselYu/UniKP/resolve/main/UniKP%20for%20kcat_Km.pkl': 'md5:bc598e880e0893bf25f8bfb27074ccac'
    }

CUSTOMIZED_PROT_T5_XL_UNIREF50_WEIGHT=os.getenv('PROT_T5_XL_UNIREF50_WEIGHT')


if CUSTOMIZED_PROT_T5_XL_UNIREF50_WEIGHT:
    if not os.path.exists(CUSTOMIZED_PROT_T5_XL_UNIREF50_WEIGHT):
        os.makedirs(CUSTOMIZED_PROT_T5_XL_UNIREF50_WEIGHT,exist_ok=True)
    DEFAULT_PROT_T5_XL_UNIREF50_WEIGHT = CUSTOMIZED_PROT_T5_XL_UNIREF50_WEIGHT
else:
    DEFAULT_PROT_T5_XL_UNIREF50_WEIGHT = os.path.join(dir_path, 'weights', 'prot_t5_xl_uniref50')

CUSTOMIZED_WEIGHTS_DIR=os.getenv('UNIKP_PRETRAINED_WEIGHT')
if CUSTOMIZED_WEIGHTS_DIR:
    if not os.path.exists(CUSTOMIZED_WEIGHTS_DIR):
        os.makedirs(CUSTOMIZED_WEIGHTS_DIR,exist_ok=True)
    DEFAULT_UNIKP_WEIGHT=CUSTOMIZED_WEIGHTS_DIR
else:
    DEFAULT_UNIKP_WEIGHT = os.path.join(dir_path, 'weights', 'UniKP')

class FileDownloader:
    def __init__(self, url, save_dir, md5sum=None):
        self.url = url
        self.save_dir = save_dir
        self.md5sum = md5sum

    def download_file(self):
        try:
            downloaded_file = self._download()
            if downloaded_file:
                if downloaded_file.lower().endswith('.zip'):
                    self._extract_zip(downloaded_file)
            else:
                return f"Failed to download file from {self.url}"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def _download(self):
        _basename=os.path.basename(self.url).split('?')[0]
        downloaded=pooch.retrieve(
            url=self.url,
            known_hash=self.md5sum,
            fname=_basename.replace('%20', '_'),
            path=self.save_dir,
            progressbar=True)
        return downloaded

    def _extract_zip(self, file_name):
        unzip_dir = os.path.splitext(file_name)[0]
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        os.remove(file_name)
        return f"File downloaded, extracted, and ZIP archive removed from {unzip_dir}"

def fetch_weights():

    # fetch prot t5 xl uniref50
    if not os.path.exists(os.path.join(DEFAULT_PROT_T5_XL_UNIREF50_WEIGHT,'prot_t5_xl_uniref50','prot_t5_xl_uniref50', 'pytorch_model.bin')):
        _basename=os.path.basename(DEFAULT_PROT_T5_XL_UNI_REF50_WEIGHT_URL)
        print(f'Downloading {_basename} ...')
        downloader=FileDownloader(
            url=DEFAULT_PROT_T5_XL_UNI_REF50_WEIGHT_URL, 
            save_dir=DEFAULT_PROT_T5_XL_UNIREF50_WEIGHT,
            md5sum=DEFAULT_PROT_T5_XL_UNI_REF50_WEIGHT_MD5
        )
        downloader.download_file()

    # fetch unikp weights
    for url,md5 in DEFAULT_UNIKP_WEIGTHS_URL.items():
        _basename=os.path.basename(url).replace('%20', '_')
        if os.path.exists(os.path.join(DEFAULT_UNIKP_WEIGHT,_basename)):
            continue
        
        print(f'Downloading {_basename} ...')
        downloader=FileDownloader(
            url=f'{url}?download=true',
            md5sum=md5,
            save_dir=DEFAULT_UNIKP_WEIGHT
        )
        downloader.download_file()
        

def split_refactored(sm): # ChatGPT version
    """
    Split SMILES into words. Handles specific chemical elements or characters.
    
    Args:
    - sm (str): A SMILES string
    
    Returns:
    - str: A string with space-separated words
    """
    special_cases = {'Cl', 'Br', 'Ca', 'Cu', 'Be', 'Ba', 'Bi', 'Si', 'Se', 'Sr',
                     'Na', 'Ni', 'Rb', 'Ra', 'Xe', 'Li', 'Al', 'As', 'Ag', 'Au',
                     'Mg', 'Mn', 'Te', 'Zn', 'Si', 'Se', 'Te', 'He', '+2', '+3',
                     '+4', '-2', '-3', '-4', 'Kr', 'Fe'}

    i = 0
    arr = []
    while i < len(sm):
        found = False
        for length in range(3, 1, -1):  # Check for special cases with length > 1
            if sm[i:i + length] in special_cases:
                arr.append(sm[i:i + length])
                i += length
                found = True
                break
        if not found:
            arr.append(sm[i])
            i += 1

    return ' '.join(arr)

# Split SMILES into words
def split(sm):
    '''
    function: Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
    input: A SMILES
    output: A string with space between words
    '''
    arr = []
    i = 0
    while i < len(sm)-1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', \
                        'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
            arr.append(sm[i])
            i += 1
        elif sm[i]=='%':
            arr.append(sm[i:i+3])
            i += 3
        elif sm[i]=='C' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='b':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='X' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='L' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='s':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='T' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='Z' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='t' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='H' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='K' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='F' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm)-1:
        arr.append(sm[i])
    return ' '.join(arr) 

# 活性化関数
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# 位置情報を考慮したFFN
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
# 正規化層
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Sample SMILES from probablistic distribution
def sample(msms):
    ret = []
    for msm in msms:
        ret.append(torch.multinomial(msm.exp(), 1).squeeze())
    return torch.stack(ret)

def validity(smiles):
    loss = 0
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            loss += 1
    return 1-loss/len(smiles)



def smiles_to_vec(Smiles, device=torch.device('cpu')):
    from UniKP.build_vocab import WordVocab
    from UniKP.pretrain_trfm import TrfmSeq2seq
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab(os.path.join(dir_path,'vocab.pkl'))
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            # print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load(os.path.join(dir_path,'trfm_12_23000.pkl'),map_location=device))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(
        Sequence, 
        prot_t5_xl_uniref50=os.path.join(DEFAULT_PROT_T5_XL_UNIREF50_WEIGHT,'prot_t5_xl_uniref50','prot_t5_xl_uniref50'),
        device=torch.device('cpu')
        ):
    from transformers import T5EncoderModel, T5Tokenizer
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    tokenizer = T5Tokenizer.from_pretrained(prot_t5_xl_uniref50, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(prot_t5_xl_uniref50)
    gc.collect()
    
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize


def get_lds_kernel_window(kernel, ks, sigma):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal.windows import triang
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window

def device_picker(device: str = ''):
    if (device.startswith('cuda') or not device) and torch.cuda.is_available():
        print('Use CUDA')
        device = torch.device('cuda')
    elif (device.startswith('mps') or not device) and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print('Use MPS')
        device = torch.device("mps")
    else:
        print('Use CPU')
        device = torch.device("cpu")
    
    return device

def save_models(Ifeature, Label, round=5, model_label='kcat',save_dir='.',ncpu=os.cpu_count()):
    for i in range(round):
        model = ExtraTreesRegressor()
        print(f'Fitting at #{i} round...')
        with parallel_backend('loky', n_jobs=ncpu):
            model.fit(Ifeature, Label)
            with open(os.path.join(save_dir,f'{model_label}_{str(i)}_model.pkl'), "wb") as f:
                pickle.dump(model, f)

if __name__ == '__main__':
    fetch_weights()
