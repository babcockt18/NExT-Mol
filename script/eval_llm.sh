#!/bin/bash


{
# export NCCL_IB_DISABLE=1 # for multi-gpu training on Song2
export CUDA_VISIBLE_DEVICES="7"
python llm_train.py  \
    --devices "[0]" \
    --mode eval \
    --init_checkpoint "all_checkpoints/llm_train_restricted_rand/epoch=89.ckpt" \
    --dataset "GeomDrugs" \
    --root "./data/drugs" \
    --llm_tune full  \
    --filename 'llm_train_restricted_rand_eval' \
    --generate_eval_epoch 5 \
    --llm_model 'mollama_ckpt/mollama_iter-2220000-ckpt_hf' \
    --rand_smiles unrestricted \
    --temperature 1.0 \
    --num_beams 1 \
    --sample_num 10000 \
    --batch_size 64 \
    --accumulate_grad_batches 4

exit
}