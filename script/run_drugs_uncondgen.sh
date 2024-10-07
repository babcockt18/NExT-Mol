#!/bin/bash
{
# export CUDA_VISIBLE_DEVICES="6";
filename="drugs_uncondgen";
ckpt_path="all_checkpoints/${filename}/last.ckpt";
if [ ! -d "$ckpt_path" ]; then
    delta_path="all_checkpoints/${filename}/delta.ckpt";
    if [ ! -d "$delta_path" ]; then
        ckpt_path="None"
    else
        ckpt_path=$delta_path
    fi        
fi

python train_uncond_gene.py  \
    --dataset "Geom-drugs-jodo" \
    --root './data/archive/jodo_data/geom' \
    --filename $filename \
    --llm_model 'acharkq/MoLlama' \
    --num_workers 4 \
    --batch_size 128 \
    --in_node_features 74 \
    --max_epochs 40000 \
    --save_every_n_epochs 50 \
    --ckpt_path ${ckpt_path} \
    --check_val_every_n_epoch 20 \
    --accumulate_grad_batches 2 \
    --not_pair_update \
    --fuse_qkv \
    --dropout 0.05 \
    --test_conform_epoch 100000 --conform_eval_epoch 100000 \
    --cache_epoch 2 \
    --generate_eval_epoch 100 \
    --eval_smiles_path "all_checkpoints/drugs_llm_e200_new/lightning_logs/version_4/smiles_epoch19.txt" \
    --mode train \
    # --diff_ckpt "all_checkpoints/trained_ckpt/dgt_drugs_small_e2999.ckpt"
exit
}