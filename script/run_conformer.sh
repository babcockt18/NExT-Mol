#!/bin/bash

# Usage
# sh ./script/run_conformer.sh --dataset=GeomDrugs --gpu=[7]
# nohup sh ./script/run_conformer.sh --dataset=GeomDrugs --gpu=[7] > ./script/nohup_output/drugs.log 2>&1 &

# Default
dataset="GeomDrugs"
gpu="0,7"

for arg in "$@"
do
    case $arg in
        --dataset=*)
        dataset="${arg#*=}"
        shift
        ;;
        --gpu=*)
        gpu="${arg#*=}"
        shift
        ;;
        *)
        echo "Unknown argument: $arg"
        exit 1
        ;;
    esac
done

# 根据不同的数据集设置参数
if [ "$dataset" = "QM9" ]; then
    root_dir="./data/qm9v6"
    unimol_path="unimol_ckpt/qm9_220908.pt"
    filename="qm9"
elif [ "$dataset" = "GeomDrugs" ]; then
    root_dir="./data/drugs"
    unimol_path="unimol_ckpt/drugs_220908.pt"
    filename="drugs"
else
    echo "Unknown dataset: $dataset"
    exit 1
fi

# llm_model="acharkq/MoLlama"
llm_model="mollama_ckpt/mollama_iter-2220000-ckpt_hf"

llm_tune="full"

filename="${filename}_${llm_tune}_tune"

filename="${filename}_delta_train"

unimol_version="v3"
filename="${filename}_unimol${unimol_version}"

# filename="${filename}_biattend"

{
# export NCCL_IB_DISABLE=1 # for multi-gpu training on Song2
export CUDA_VISIBLE_DEVICES="${gpu}"
python uncond_generate.py \
    --devices "auto" \
    --dataset "$dataset" \
    --root "$root_dir" \
    --llm_model "$llm_model" \
    --llm_tune "$llm_tune" \
    --unimol_path "$unimol_path" \
    --unimol_version "$unimol_version" \
    --unimol_distance_from_coord \
    --lm_loss 0 \
    --check_val_every_n_epoch 10 \
    --save_every_n_epochs 5 \
    --generate_eval_epoch 100 \
    --filename "$filename"  \
    --batch_size 64 \
    --delta_train \
    --temperature 1.0 \
    --num_beams 1 \
    --num_workers 1 \
    # --init_checkpoint "" \
    # --bi_attend

exit
}