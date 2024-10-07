# 1D to 3D Molecular Conformer Generation

## New Environment Setup (without flash attention, e3nn, and UniMol)

- conda create -n 123d python=3.8
- conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
- conda install pandas
- conda install pyg pytorch-cluster pytorch-scatter -c pyg
- pip install setuptools==69.5.1
- pip install transformers lightning deepspeed rdkit nltk rouge_score peft selfies scikit-learn-extra chardet
- pip install fcd_torch pomegranate
- pip install git+https://github.com/acharkq/moses


## Environment Setup

- conda create -n 123d python=3.8
- conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
- conda install nvidia/label/cuda-11.8.0::cuda-nvcc
- conda install -c "nvidia/label/cuda-11.8.0" cuda-libraries-dev
<!-- - pip install https://github.com/dptech-corp/Uni-Core/releases/download/0.0.3/unicore-0.0.1+cu118torch2.1.2-cp38-cp38-linux_x86_64.whl -->
- pip install git+https://github.com/dptech-corp/Uni-Core.git
- conda install pandas
- conda install pyg pytorch-cluster pytorch-scatter -c pyg
- pip install flash-attn --no-build-isolation
- pip install transformers lightning deepspeed rdkit nltk rouge_score peft selfies scikit-learn-extra chardet
- pip install fcd_torch pomegranate e3nn lmdb
- pip install git+https://github.com/acharkq/moses

## Clone This Project

We use git-lfs to store some datasets and checkpoints. Therefore, make sure git-lfs is installed before cloning this git repository

```sh
conda install git # if you do not have git already
conda install git-lfs
git-lfs install
git clone url-to-this-project
```

## Dataset

Processed QM9 dataset [link](https://drive.google.com/file/d/1mu6qicmCzqJ6WosJ55n0VUywzo598925/view?usp=sharing)


## some script

```
# evaluate qm9 3d unconditional generation

python evaluate.py --root './data/qm9v7'  --dataset QM9-jodo --path all_checkpoints/qm9_uncondgen/lightning_logs/version_3/predictions_0.pt
```