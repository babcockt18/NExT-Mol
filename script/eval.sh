#!/bin/bash

# Usage
# sh ./script/eval_conformer.sh --dataset=GeomDrugs --gpu=[7]
# nohup sh ./script/eval_conformer.sh --dataset=GeomDrugs --gpu=[7] > ./script/nohup_output/drugs.log 2>&1 &

# Default
dataset="GeomDrugs"
gpu="0"
init_checkpoint="/home/luoyc/123D/all_checkpoints/drugs_full_tune_delta_train_unimolv3/epoch=99.ckpt"

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
        --init_checkpoint=*)
        init_checkpoint="${arg#*=}"
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

llm_tune="full"
# llm_model="acharkq/MoLlama"
llm_model="mollama_ckpt/mollama_iter-2220000-ckpt_hf"

filename="${filename}_${llm_tune}_tune"

# delta_train=""
delta_train="--delta_train"
filename="${filename}_delta_train"

unimol_version="v3"
filename="${filename}_unimol${unimol_version}"

lm_loss=0

check_val_every_n_epoch=10
save_every_n_epochs=5
generate_eval_epoch=200

bi_attend=""
# bi_attend="--bi_attend"
# filename="${filename}_biattend"

sample_num=1000

{
export CUDA_VISIBLE_DEVICES="${gpu}"
python uncond_generate.py \
    --devices "-1" \
    --dataset "$dataset" \
    --root "$root_dir" \
    --llm_model "$llm_model" \
    --llm_tune "$llm_tune" \
    --unimol_path "$unimol_path" \
    --unimol_version "$unimol_version" \
    --unimol_distance_from_coord \
    --lm_loss "$lm_loss" \
    --check_val_every_n_epoch "$check_val_every_n_epoch" \
    --save_every_n_epochs "$save_every_n_epochs" \
    --generate_eval_epoch "$generate_eval_epoch" \
    --filename "$filename"  \
    --mode eval_conf \
    --init_checkpoint "$init_checkpoint" \
    --sample_num "$sample_num" \
    $delta_train \
    $bi_attend

exit
}
