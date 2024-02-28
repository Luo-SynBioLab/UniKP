<!-- Add banner here -->

# UniKP

<!-- Add buttons here -->
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
![GitHub last commit](https://img.shields.io/badge/Last%20commit-May-critical)
![GitHub issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
![GitHub pull requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)
![GitHub](https://img.shields.io/badge/license-gpl--3.0-informational)

<!-- Describe your project in brief -->
**Feel free to contact me via email at yuhanid147@gmail.com if you encounter any issues or have any questions.**

**Introduction of UniKP.**

Prediction of enzyme kinetic parameters is essential for designing and optimizing enzymes for various biotechnological and industrial applications, but the limited performance of current prediction tools on diverse tasks hinders their practical applications. Here, we introduce UniKP, a unified framework based on pretrained language models for the prediction of enzyme kinetic parameters, including enzyme turnover number (*k*<sub>cat</sub>), Michaelis constant (*K*<sub>m</sub>), and catalytic efficiency (*k*<sub>cat</sub> / *K*<sub>m</sub>), from protein sequences and substrate structures. A two-layer framework derived from UniKP (EF-UniKP) has also been proposed to allow robust *k*<sub>cat</sub> prediction in considering environmental factors, including pH and temperature. In addition, four representative re-weighting methods are systematically explored to successfully reduce the prediction error in high-value prediction tasks. We have demonstrated the application of UniKP and EF-UniKP in several enzyme discovery and directed evolution tasks, leading to the identification of new enzymes and enzyme mutants with higher activity. UniKP is a valuable tool for deciphering the mechanisms of enzyme kinetics and enables novel insights into enzyme engineering and their industrial applications.

**Here is the framework of UniKP.**
<p align="center">
  <img  src="Figures/UniKP.png" >
</p>

# Demo-Preview

- **For users who want to know what to expect in this project, as follows:**

  - (1). Out the *k*<sub>cat</sub> values given protein sequences and substrate structures.
  - (2). Out the *K*<sub>m</sub> values given protein sequences and substrate structures.
  - (3). Out the *k*<sub>cat</sub> / *K*<sub>m</sub> values given protein sequences and substrate structures.

|Input_v1|Input_v2|Model|Output|
|--|--|--|--|
| MSELMKLSAV...MAQR | CC(O)O | UniKP for *k*<sub>cat</sub> | 2.75 s<sup>-1</sup> |
| MSELMKLSAV...MAQR | CC(O)O | UniKP for *K*<sub>m</sub> |  0.36 mM |
| MSELMKLSAV...MAQR | CC(O)O | UniKP for *k*<sub>cat</sub> / *K*<sub>m</sub> |  9.51 s<sup>-1</sup> * mM<sup>-1</sup> |

# Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [UniKP](#unikp)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Development](#development)
- [Contribute](#contribute)
    - [Sponsor](#sponsor)
    - [Adding new features or fixing bugs](#adding-new-features-or-fixing-bugs)
- [License](#license)
- [Footer](#footer)

# Prerequisites
[(Back to top)](#table-of-contents)

Notice:
- You need download pretrained protein language modoel *ProtT5-XL-UniRef50* to generate enzyme representation, the link is provided on [ProtT5-XL-U50](https://zenodo.org/records/4644188).
- You also need download model *UniKP* for ***k*<sub>cat</sub>, *K*<sub>m</sub>** and ***k*<sub>cat</sub> / *K*<sub>m</sub>** to predict corresponding kinetic parameters, the link is provided on [UniKP_model](https://huggingface.co/HanselYu/UniKP/tree/main).

**Place these two downloaded models in the UniKP directory.**

- We have included pretrained molecular language modoel *SMILES Transformer* in this repository to generate substrate representation, the link is also provided on [SMILES Transformer](https://github.com/DSPsleeporg/smiles-transformer).

<!-- *You might have noticed the **Back to top** button(if not, please notice, it's right there!). This is a good idea because it makes your README **easy to navigate.*** 

The first one should be how to install(how to generally use your project or set-up for editing in their machine).

This should give the users a concrete idea with instructions on how they can use your project repo with all the steps.

Following this steps, **they should be able to run this in their device.**

A method I use is after completing the README, I go through the instructions from scratch and check if it is working. -->

<!-- Here is a sample instruction:

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone https://github.com/navendu-pottekkat/nsfw-filter.git``` -->

# Usage
[(Back to top)](#table-of-contents)
- **For users who want to use the deep learning model for prediction, please run these command lines at the terminal:**
  - (1). Download the UniKP package
  
         git clone https://github.com/Luo-SynBioLab/UniKP
  
  - (2). Create and activate enviroment
  
         conda create -n Uni_test python=3.7
         conda activate Uni_test

  - (3). Download required Python package
         
         cd UniKP
         pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
         pip install -r requirements.txt   

- Example for how to predict enzyme kinetic parameters from enzyme sequences and substrate structures by language model, UniKP:

**All predicted values have been logarithmically transformed with a base of 10. Remember to revert the transformation.**

```
import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
# build_vocab, pretrain_trfm, utils packages are from SMILES Transformer
from transformers import T5EncoderModel, T5Tokenizer
# transformers package is from ProtTrans
import re
import gc
import numpy as np
import pandas as pd
import pickle
import math


def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            print('SMILES is too long ({:d})'.format(len(sm)))
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
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(Sequence):
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
    ###### you should place downloaded model into this directory.
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    features_array = np.array(features)
    for n in range(len(features)):
        features_normalize[n] = features_array[n].mean(axis=0, dtype=np.float64)
    return features_normalize


if __name__ == '__main__':
    sequences = ['MEDIPDTSRPPLKYVKGIPLIKYFAEALESLQDFQAQPDDLLISTYPKSGTTWVSEILDMIYQDGDVEKCRRAPVFIRVPFLEFKA'
                 'PGIPTGLEVLKDTPAPRLIKTHLPLALLPQTLLDQKVKVVYVARNAKDVAVSYYHFYRMAKVHPDPDTWDSFLEKFMAGEVSYGSW'
                 'YQHVQEWWELSHTHPVLYLFYEDMKENPKREIQKILKFVGRSLPEETVDLIVQHTSFKEMKNNSMANYTTLSPDIMDHSISAFMRK'
                 'GISGDWKTTFTVAQNERFDADYAKKMEGCGLSFRTQL']
    Smiles = ['OC1=CC=C(C[C@@H](C(O)=O)N)C=C1']
    seq_vec = Seq_to_vec(sequences)
    smiles_vec = smiles_to_vec(Smiles)
    fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)

    ###### you should place downloaded model into this directory.
    # For kcat
    with open('UniKP/UniKP for kcat.pkl', "rb") as f:
        model = pickle.load(f)
    # For Km
    # with open('UniKP/UniKP for Km.pkl', "rb") as f:
    #     model = pickle.load(f)
    # For kcat/Km
    # with open('UniKP/UniKP for kcat_Km.pkl', "rb") as f:
    #     model = pickle.load(f)
    
    Pre_label = model.predict(fused_vector)
    Pre_label_pow = [math.pow(10, Pre_label[i]) for i in range(len(Pre_label))]
    print(len(Pre_label_pow))
    res = pd.DataFrame({'sequences': sequences, 'Smiles': Smiles, 'Pre_label': Pre_label_pow})
    res.to_excel('Kinetic_parameters_predicted_label.xlsx')
```

<!-- This is optional and it is used to give the user info on how to use the project after installation. This could be added in the Installation section also. -->

# Development
[(Back to top)](#table-of-contents)

<!-- This is the place where you give instructions to developers on how to modify the code.

You could give **instructions in depth** of **how the code works** and how everything is put together.

You could also give specific instructions to how they can setup their development environment.

Ideally, you should keep the README simple. If you need to add more complex explanations, use a wiki. Check out [this wiki](https://github.com/navendu-pottekkat/nsfw-filter/wiki) for inspiration. -->

# Contribute
[(Back to top)](#table-of-contents)

 * <b>Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen 518055, China:</b><br/>
 
| Han Yu       |      Huaxiang Deng  |  Jiahui He| Jay D. Keasling | Xiaozhou Luo |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
<!-- | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/ElnaggarAhmend.jpg?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/MichaelHeinzinger-2.jpg?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/christiandallago.png?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/female.png?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/B.Rost.jpg?raw=true"> | -->

<!-- This is where you can let people know how they can **contribute** to your project. Some of the ways are given below.

Also this shows how you can add subsections within a section. -->

### Sponsor
[(Back to top)](#table-of-contents)

We would like to acknowledge the support from National Key R&D Program of China (2018YFA0903200), National Natural Science Foundation of China (32071421), Guangdong Basic and Applied Basic Research Foundation (2021B1515020049), Shenzhen Science and Technology Program (ZDSYS20210623091810032 and JCYJ20220531100207017), and Shenzhen Institute of Synthetic Biology Scientific Research Program (ZTXM20203001).

<!-- Your project is gaining traction and it is being used by thousands of people(***with this README there will be even more***). Now it would be a good time to look for people or organisations to sponsor your project. This could be because you are not generating any revenue from your project and you require money for keeping the project alive.

You could add how people can sponsor your project in this section. Add your patreon or GitHub sponsor link here for easy access.

A good idea is to also display the sponsors with their organisation logos or badges to show them your love!(*Someday I will get a sponsor and I can show my love*) -->

### Adding new features or fixing bugs
[(Back to top)](#table-of-contents)

<!-- This is to give people an idea how they can raise issues or feature requests in your projects. 

You could also give guidelines for submitting and issue or a pull request to your project.

Personally and by standard, you should use a [issue template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/ISSUE_TEMPLATE.md) and a [pull request template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/PULL_REQ_TEMPLATE.md)(click for examples) so that when a user opens a new issue they could easily format it as per your project guidelines.

You could also add contact details for people to get in touch with you regarding your project. -->

# License
[(Back to top)](#table-of-contents)

<!-- Adding the license to README is a good practice so that people can easily refer to it.

Make sure you have added a LICENSE file in your project folder. **Shortcut:** Click add new file in your root of your repo in GitHub > Set file name to LICENSE > GitHub shows LICENSE templates > Choose the one that best suits your project!

I personally add the name of the license and provide a link to it like below. -->

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)

# Footer
[(Back to top)](#table-of-contents)

If you use this code or our models for your publication, please cite the original paper:

Yu, H., Deng, H., He, J. et al. UniKP: a unified framework for the prediction of enzyme kinetic parameters. Nat Commun 14, 8211 (2023). [https://doi.org/10.1038/s41467-023-44113-1]

The preprint version:

Han Yu, Huaxiang Deng, Jiahui He et al. Highly accurate enzyme turnover number prediction and enzyme engineering with PreKcat, 18 May 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-2749688/v1]

<!-- Let's also add a footer because I love footers and also you **can** use this to convey important info.

Let's make it an image because by now you have realised that multimedia in images == cool(*please notice the subtle programming joke). -->
<!-- 
Leave a star in GitHub, give a clap in Medium and share this guide if you found this helpful. -->

<!-- Add the footer here -->

<!-- ![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) -->
