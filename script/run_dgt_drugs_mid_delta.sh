{
# export CUDA_VISIBLE_DEVICES="7";

filename="dgt_drugs_mid_delta";
ckpt_path="all_checkpoints/${filename}/last.ckpt";

if [ ! -d "$ckpt_path" ]; then
    delta_path="all_checkpoints/${filename}/delta.ckpt";
    if [ ! -d "$delta_path" ]; then
        ckpt_path="None"
    else
        ckpt_path=$delta_path
    fi        
fi

python train_lm_conf.py --dataset "Geom-drugs-df" --root "./data/tordf_drugs_v2" --in_node_features 74 --min_lr 1e-5 --init_lr 0.0001 --warmup_lr 1e-6 --warmup_steps 1000 --batch_size 32 --num_workers 2 --max_epochs 16000 --mode train --save_every_n_epochs 100 --ckpt_path ${ckpt_path} --n_blocks 12 --hidden_size 768 --dropout 0.1 \
--filename $filename  --llm_model "/home/zhiyuan/hf_repo/models--acharkq--MoLlama/snapshots/e94dd9943e0620313b693d92149ed8bd7a70bb96"  --test_conform_epoch 100000 --conform_eval_epoch 100000 --check_val_every_n_epoch 20 --generate_eval_epoch 100000 --sampling_steps 100 --rand_smiles canonical --num_nodes 2 --world_size 16 --cache_epoch 1  --use_llm  --delta_train --rand_smiles canonical --use_self_att_proj --llm_jk mean --use_llm_projector --llm_tune lora --tune_embedding --max_epochs 5000 --accumulate_grad_batches 1 --diff_ckpt "all_checkpoints/dgt_drugs_mid_v4_ema/epoch=2999.ckpt/converted.ckpt" --use_flash_attention;

# python train_lm_conf.py --dataset "Geom-drugs-df" --root "./data/tordf_drugs_v2" --in_node_features 74 --min_lr 1e-5 --init_lr 0.0001 --warmup_lr 1e-6 --warmup_steps 1000 --batch_size 32 --num_workers 2 --max_epochs 16000 --mode train --save_every_n_epochs 100 --ckpt_path ${ckpt_path} --n_blocks 12 --hidden_size 768 --dropout 0.1 \
# --filename $filename  --llm_model "acharkq/MoLlama"  --test_conform_epoch 100000 --conform_eval_epoch 100000 --check_val_every_n_epoch 20 --generate_eval_epoch 100000 --sampling_steps 100 --rand_smiles canonical --num_nodes 1 --world_size 16 --cache_epoch 1  --use_llm  --delta_train --rand_smiles canonical --use_self_att_proj --llm_jk mean --use_llm_projector --llm_tune lora --tune_embedding --max_epochs 5000 --accumulate_grad_batches 1 --diff_ckpt "all_checkpoints/trained_ckpt/dgt_drugs_mid_v4_ema_e2999.ckpt" --use_flash_attention;
exit
}