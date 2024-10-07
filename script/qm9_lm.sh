{
export CUDA_VISIBLE_DEVICES='0,1';
python llm_train.py  \
    --dataset "QM9" \
    --root "./data/qm9v6" \
    --llm_tune full  \
    --filename 'llm_train_restricted_rand' \
    --generate_eval_epoch 5 \
    --llm_model 'acharkq/MoLlama' \
    --rand_smiles restricted \
    --temperature 1.0 \
    --num_beams 1 \
    --sample_num 10000 \
    --batch_size 64 \
    --accumulate_grad_batches 2
}