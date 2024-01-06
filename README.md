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
  - (1). Create and activate enviroment
  
         conda create -n UniKP 'python>=3.9,<3.12' -y
         conda activate UniKP
         git clone https://github.com/YaoYinYing/UniKP.git
         git checkout pip-installable

  - (2). Install UniKP
         
         pip install git+https://github.com/YaoYinYing/UniKP.git@pip-installable

  - (3). Fetch ProtT5 XL weights and UniKP pretrained weight (original weights provied by the authors)
  
         # set costomized download path (optional. If set, write these statements into shell profile like `.bashrc` or `.zshrc`)
         export PROT_T5_XL_UNIREF50_WEIGHT=/path/to/ml/weights/
         export UNIKP_PRETRAINED_WEIGHT=/path/to/ml/weights/

         # download all models for 
         python $CONDA_PREFIX/lib/python<python-version>/site-packages/UniKP/utils.py
         
         

- Example for how to predict enzyme kinetic parameters from enzyme sequences and substrate structures by language model, UniKP:

**All predicted values have been logarithmically transformed with a base of 10. Remember to revert the transformation.**

```bash
# get help message
python /repo/UniKP/examples/inference.py --help
Usage: inference.py [OPTIONS]

Options:
  -s, --sequence TEXT             Protein Squence input  [required]
  -m, --smiles TEXT               SMILES input  [required]
  -w, --weight [Km|Kcat|Kcat_Km|All]
                                  Weight model, `Km`, `Kcat`, `Kcat_Km` or
                                  `All` for all.  [required]
  -p, --prefix TEXT               Prefix label of this prediction
  -o, --output_dir TEXT           Output directory
  -d, --device TEXT               Device to use. `cuda` for CUDA, `mps` for
                                  Apple Silicon MPS, `cpu` for pure CPU. If
                                  unset, device is picked automatically.
  -j, --nprocs INTEGER            Number of processors
  --help                          Show this message and exit.
```
```bash
python /repo/UniKP/examples/inference.py -s 'MEDIPDTSRPPLKYVKGIPLIKYFAEALESLQDFQAQPDDLLISTYPKSGTTWVSEILDMIYQDGDVEKCRRAPVFIRVPFLEFKAPGIPTGLEVLKDTPAPRLIKTHLPLALLPQTLLDQKVKVVYVARNAKDVAVSYYHFYRMAKVHPDPDTWDSFLEKFMAGEVSYGSWYQHVQEWWELSHTHPVLYLFYEDMKENPKREIQKILKFVGRSLPEETVDLIVQHTSFKEMKNNSMANYTTLSPDIMDHSISAFMRKGISGDWKTTFTVAQNERFDADYAKKMEGCGLSFRTQL' -m 'OC1=CC=C(C[C@@H](C(O)=O)N)C=C1' -w 'Kcat' -d cuda
```

- Retraining:
  
This takes long time, depending on the device you are using.
```bash
bash /repo/UniKP/scripts/retraining.sh

cp /repo/UniKP/retrained/Km_0_model.pkl $UNIKP_PRETRAINED_WEIGHT/UniKP_for_Km.pkl
cp /repo/UniKP/retrained/kcat_0_model.pkl $UNIKP_PRETRAINED_WEIGHT/UniKP_for_kcat.pkl
cp /repo/UniKP/retrained/Kcat_Km_0_model.pkl $UNIKP_PRETRAINED_WEIGHT/UniKP_for_kcat_Km.pkl
```

- Fine-tuning:
**Work in Progress**


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
