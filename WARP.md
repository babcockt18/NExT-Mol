# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n nextmol python=3.8
conda activate nextmol

# Install PyTorch and dependencies
conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas
conda install pyg pytorch-cluster pytorch-scatter -c pyg

# Install additional packages
pip install setuptools==69.5.1
pip install transformers lightning deepspeed rdkit nltk rouge_score peft selfies scikit-learn-extra chardet
pip install fcd_torch pomegranate
pip install git+https://github.com/molecularsets/moses
```

### LLM Training and 1D Molecule Generation

```bash
# Train LLM on QM9 dataset (for de novo generation)
python llm_train.py \
    --dataset "QM9" \
    --root <path-to-QM92014> \
    --llm_tune full \
    --filename 'qm9_llm' \
    --generate_eval_epoch 5 \
    --llm_model "acharkq/MoLlama" \
    --rand_smiles restricted \
    --temperature 1.0 \
    --num_beams 1 \
    --sample_num 10000 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --max_epochs 100

# Train LLM for conditional generation (specify property)
condition="mu"  # Options: 'mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv'
python llm_train.py \
    --dataset "QM9" \
    --root <path-to-QM92014> \
    --llm_tune full \
    --filename "qm9_llm_conditional_${condition}" \
    --llm_model "acharkq/MoLlama" \
    --condition_property "${condition}" \
    --max_epochs 100
```

- Generated SMILES files will be saved in the log folder after training
- Use `--dataset "GeomDrugs-JODO"` for GEOM-DRUGS dataset
- Add `--use_flash_attention` for better memory efficiency

### Diffusion Model Training for 3D Conformer Generation

```bash
# Train DMT model for conformer prediction (QM9)
python train_lm_conf.py \
    --dataset "QM9-df" \
    --root <path-to-GEOM-QM9> \
    --batch_size 256 \
    --num_workers 4 \
    --max_epochs 2000 \
    --mode train \
    --filename 'dmt-b-qm9' \
    --llm_model "acharkq/MoLlama" \
    --sampling_steps 100

# Integrate with MoLlama (Stage 2 & 3)
python train_lm_conf.py \
    --dataset "QM9-df" \
    --root <path-to-GEOM-QM9> \
    --filename "dmt-b-llm-qm9" \
    --llm_model "acharkq/MoLlama" \
    --use_llm \
    --delta_train \
    --llm_tune lora \
    --tune_embedding \
    --use_flash_attention \
    --diff_ckpt <path-to-dmt-checkpoint>
```

- For GEOM-DRUGS, use `--dataset "Geom-drugs-df"` and adjust `--in_node_features 74`
- For DMT-L model, add `--n_blocks 12 --hidden_size 768`
- Use `--world_size 8` for multi-GPU training with 8 GPUs

### Unconditional 3D Molecule Generation

```bash
# Train diffusion model for de novo generation
python train_uncond_gene.py \
    --dataset "QM9-jodo" \
    --root <path-to-QM92014> \
    --filename "qm9_denovo_gen" \
    --llm_model "acharkq/MoLlama" \
    --batch_size 128 \
    --max_epochs 40000 \
    --save_every_n_epochs 50 \
    --generate_eval_epoch 100 \
    --eval_smiles_path <path-to-generated-smiles> \
    --mode train
```

### Evaluation Commands

```bash
# Evaluate conformer prediction
python train_lm_conf.py \
    --dataset "QM9-df" \
    --root <path-to-GEOM-QM9> \
    --mode eval_test_conform \
    --filename 'eval_dmt_b_qm9' \
    --llm_model "acharkq/MoLlama" \
    --init_checkpoint <checkpoint-path> \
    --infer_batch_size 512 \
    --dropout 0.05

# Evaluate conformer metrics from pickle file
python eval_confs.py \
    --input <path-to-predict.pkl> \
    --dataset Geom-drugs-df \
    --root <path-to-GEOM-DRUGS>

# Evaluate generated molecules (2D metrics)
python evaluate.py \
    --path <path-to-smiles-file> \
    --mode smiles \
    --dataset GeomDrugs-JODO \
    --root <path-to-dataset>

# Evaluate generated molecules (3D metrics)
python evaluate.py \
    --path <path-to-3dmol-file> \
    --mode 3dmol \
    --dataset QM9-jodo \
    --root <path-to-dataset>
```

### Checkpoint Management

- Checkpoints are saved in `all_checkpoints/<filename>/`
- Use `--init_checkpoint <path>` to load a specific checkpoint
- Use `--ckpt_path <path>` to resume training from a checkpoint
- Last checkpoint is linked as `last<epoch>.ckpt`
- EMA checkpoints available when training with `--use_ema`

## Repository & Training Pipeline Overview

NExT-Mol implements a two-stage molecular generation pipeline combining language models with 3D diffusion. The architecture consists of three main training scripts (`llm_train.py`, `train_lm_conf.py`, `train_uncond_gene.py`) that orchestrate PyTorch Lightning modules for different generation tasks.

The `model/` directory contains the core neural network modules:
- **LLMPL** (`llm_pl.py`): Lightning wrapper for MoLlama language model that generates SMILES sequences
- **DiffusionPL** (`diffusion_pl.py`): Lightning module for conditional 3D conformer generation using denoising diffusion
- **UncondGenPL** (`uncond_gen_pl.py`): Unconditional 3D molecule generation combining LLM and diffusion
- **Equiformer modules** (`model/equiformer/`): SE(3)-equivariant transformer layers for processing 3D molecular structures
- **EMA** (`ema.py`): Exponential moving average callback for stable training

The `data_provider/` directory handles dataset-specific data loading and preprocessing:
- **QM9 variants**: `QM9DataModule`, `QM9TorDFDataModule`, `QM9DM` for different training stages
- **GEOM-DRUGS variants**: `GeomDrugsLMDataModule`, `GeomDrugsTorDFDataModule`, `GeomDrugsJODODM`
- Each datamodule manages molecule tokenization, conformer loading, and property normalization

The training pipeline follows this flow:
1. **Stage 1**: Train MoLlama to generate valid SMILES sequences (unconditional or property-conditional)
2. **Stage 2**: Train DMT diffusion model for 3D conformer prediction from SMILES
3. **Stage 3**: Fine-tune with LoRA adapters to integrate LLM representations into diffusion model

Lightning callbacks handle checkpointing with configurable save frequencies (`--save_every_n_epochs`) and automatic caching of the best model. The pipeline supports multi-GPU training via DeepSpeed Stage 2 (`--strategy_name deepspeed`) and distributed data parallel strategies. Tokenization uses the HuggingFace MoLlama tokenizer with special tokens for molecular representations. The noise scheduler and position standardization are dataset-specific and managed by the datamodules.

## Datasets

Download preprocessed datasets from the OSF link provided in the README:
- **QM92014.zip**: QM9 dataset for de novo generation
- **GEOM-QM9.zip**: QM9 with conformers for conformer prediction
- **GEOM-DRUGS.zip**: GEOM-DRUGS dataset for drug-like molecules
- **geom_drugs_jodo.zip**: GEOM-DRUGS with JODO split

Place datasets in the `data/` directory and reference with `--root` parameter.

## Pretrained Models

- MoLlama language model: Available at [HuggingFace](https://huggingface.co/acharkq/MoLlama)
- DMT-B and DMT-L checkpoints: Download from OSF link in README
- Place checkpoints in `all_checkpoints/` directory
