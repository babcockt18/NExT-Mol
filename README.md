# NEXT-MOL: 3D DIFFUSION MEETS 1D LANGUAGE MODELING FOR 3D MOLECULE GENERATION

## New Environment Setup

- conda create -n nextmol python=3.8
- conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
- conda install pandas
- conda install pyg pytorch-cluster pytorch-scatter -c pyg
- pip install setuptools==69.5.1
- pip install transformers lightning deepspeed rdkit nltk rouge_score peft selfies scikit-learn-extra chardet
- pip install fcd_torch pomegranate
- pip install git+https://github.com/molecularsets/moses


## Pretrained Models

You can find the pretrained MoLlama model in the following link [OSF](https://osf.io/gqy39/?view_only=5905ef8957f9444a8808fd49933b35c7)

You can find the pretrained DMT-B and DMT-L checkpoints in the following link [OSF](https://osf.io/gqy39/?view_only=5905ef8957f9444a8808fd49933b35c7)

## Train from Scratch

### De novo 3D molecule generation

**QM9-2014 dataset:**


Dataset: Finding the preprocessed dataset named "QM92014.zip" in [OSF](https://osf.io/gqy39/?view_only=5905ef8957f9444a8808fd49933b35c7)

Step 1: train llm and generate 1D molecule sequences

```python
{
export CUDA_VISIBLE_DEVICES='0,1';
python llm_train.py  \
    --dataset "QM9" \
    --root path-to-QM92014 \
    --llm_tune full  \
    --filename 'qm9_llm' \
    --generate_eval_epoch 5 \
    --llm_model path-to-mollama \
    --rand_smiles restricted \
    --temperature 1.0 \
    --num_beams 1 \
    --sample_num 10000 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --max_epochs 100
}
```

After training, find the generated smiles file in the log folder.

Step 2: train diffusion model and generate 3D conformers (with 8 A100 GPUs)

```python
{
python train_uncond_gene.py  \
    --dataset "QM9-jodo" \
    --root path-to-QM92014 \
    --filename "qm9_denovo_gen" \
    --llm_model path-to-mollama \
    --num_workers 4 \
    --batch_size 128 \
    --max_epochs 40000 \
    --save_every_n_epochs 50 \
    --check_val_every_n_epoch 20 \
    --accumulate_grad_batches 2 \
    --dropout 0.05 \
    --test_conform_epoch 100000 --conform_eval_epoch 100000 \
    --cache_epoch 2 \
    --generate_eval_epoch 100 \
    --eval_smiles_path path-to-smiles \
    --mode train 
exit
}
```

**GEOM-DRUGS dataset:**

Dataset: We use the dataset split provided by [JODO](https://github.com/GRAPH-0/JODO). For our preprocessed version, find the dataset named "geom_drugs_jodo.zip" in the link [OSF](https://osf.io/gqy39/?view_only=5905ef8957f9444a8808fd49933b35c7)


Step 1: train llm and generate 1D molecule sequences

```python
{
python llm_train.py  \
    --dataset "GeomDrugs-JODO" \
    --root path-to-geom-drugs-jodo \
    --llm_tune full  \
    --filename "drugs_llm" \
    --generate_eval_epoch 10 \
    --llm_model path-to-mollama \
    --rand_smiles restricted \
    --temperature 1.0 \
    --num_beams 1 \
    --sample_num 10000 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --use_flash_attention \
}
```

After training, find the generated smiles file in the log folder.

Step 2: train diffusion model and generate 3D conformers (with 8 A100 GPUs)

```python
{
python train_uncond_gene.py  \
    --dataset "GeomDrugs-JODO" \
    --root path-to-geom-drugs-jodo \
    --filename "drugs_denovo_gen" \
    --llm_model path-to-mollama \
    --num_workers 4 \
    --batch_size 128 \
    --max_epochs 40000 \
    --save_every_n_epochs 50 \
    --check_val_every_n_epoch 20 \
    --accumulate_grad_batches 2 \
    --dropout 0.05 \
    --test_conform_epoch 100000 --conform_eval_epoch 100000 \
    --cache_epoch 2 \
    --generate_eval_epoch 100 \
    --eval_smiles_path path-to-smiles \
    --mode train 
exit
}
```

### 3D Conformer prediction


GEOM-QM9 dataset (with 8 A100 GPUs):

```python
python train_lm_conf.py --dataset "QM9-df" --root "./data/GEOM-QM9" --batch_size 256 --num_workers 4 --max_epochs 2000 --mode train --diff_version dgt --save_every_n_epochs 50  --filename 'dmt-b-qm9' --llm_model path-to-mollama  --test_conform_epoch 50 --conform_eval_epoch 50 --generate_eval_epoch 50  --sampling_steps 100
```

GEOM-DRUGS dataset (with 16 A100 GPUs) DMT-L:

```python
{

python train_lm_conf.py --dataset "Geom-drugs-df" --root "./data/tordf_drugs_v2" --in_node_features 74  --warmup_steps 1000 --batch_size 32 --num_workers 2 --max_epochs 16000 --mode train --save_every_n_epochs 200 --n_blocks 12 --hidden_size 768  --filename "dmt-l-drugs"    --llm_model path-to-mollama  --test_conform_epoch 100000 --conform_eval_epoch 100000 --check_val_every_n_epoch 20 --generate_eval_epoch 100000 --sampling_steps 100;
exit
}
```

GEOM-DRUGS dataset (with 16 A100 GPUs) DMT-B:

```python
{

python train_lm_conf.py --dataset "Geom-drugs-df" --root "./data/tordf_drugs_v2" --in_node_features 74 --warmup_steps 1000 --batch_size 32 --num_workers 2 --max_epochs 16000 --mode train --save_every_n_epochs 200 --filename "dmt-b-drugs"    --llm_model path-to-mollama  --test_conform_epoch 100000 --conform_eval_epoch 100000 --check_val_every_n_epoch 20 --generate_eval_epoch 100000 --sampling_steps 100;
exit
}
```

