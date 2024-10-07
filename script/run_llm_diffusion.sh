# nohup sh ./script/run_llm_diffusion.sh > ./script/nohup_output/qm9_llm_diffusion.log 2>&1 &
{
# export NCCL_IB_DISABLE=1 # for multi-gpu training on Song2
export CUDA_VISIBLE_DEVICES="0,1"
python uncond_generate_v2.py \
    --dataset "QM9-df" \
    --root "./data/tordf_qm9_v2" \
    --filename 'qm9_llm_diffusion_cgloss1_mid_lora' \
    --llm_model "mollama_ckpt/mollama_iter-2220000-ckpt_hf" \
    --llm_tune mid_lora \
    --rand_smiles restricted \
    --init_lr 0.0001  \
    --warmup_steps 1000  \
    --batch_size 256 \
    --test_conform_epoch 20 \
    --conform_eval_epoch 5 \
    --generate_eval_epoch 2 \
    --num_workers 2 \
    --max_epochs 10000 \
    --sampling_steps 5 \
    --num_conv_layers 8 \
    --hidden_size 256 \
    --tp_trans \
    --mode train  \
    --save_every_n_epochs 10  \
    --val_time 0.6 \
    --infer_time 0.4 \
    --pred_noise \
    --aug_rotation \
    --lm_loss 1.0 \
    --cg_loss 1.0 \

exit
}