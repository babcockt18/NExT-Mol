{
# export CUDA_VISIBLE_DEVICES="5,6";
filename="drugs_rand_llm_train_restricted_rand";
ckpt_path="all_checkpoints/${filename}/last.ckpt";
if [ ! -d "$ckpt_path" ]; then
    delta_path="all_checkpoints/${filename}/delta.ckpt";
    if [ ! -d "$delta_path" ]; then
        ckpt_path="None"
    else
        ckpt_path=$delta_path
    fi        
fi

python llm_train.py  \
    --dataset "GeomDrugs-JODO" \
    --root "./data/jodo_data/geom" \
    --llm_tune full  \
    --filename $filename \
    --generate_eval_epoch 10 \
    --llm_model 'acharkq/MoLlama' \
    --rand_smiles restricted \
    --temperature 1.0 \
    --num_beams 1 \
    --sample_num 10000 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --use_flash_attention \
    --load_random_llm
}
