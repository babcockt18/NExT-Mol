{
filename="dgt";
ckpt_path="all_checkpoints/${filename}/last.ckpt";
if [ ! -d "$ckpt_path" ]; then
        ckpt_path="None"
fi

python train_lm_conf.py --dataset "QM9-df" --root "./data/tordf_qm9_v2" --min_lr 1e-5 --init_lr 0.0001 --warmup_lr 1e-6 --warmup_steps 1000  --batch_size 256 --num_workers 4 --max_epochs 2000 --mode train --diff_version dgt --save_every_n_epochs 50 --ckpt_path ${ckpt_path} \
		--filename $filename  --llm_model "acharkq/MoLlama"  --n_blocks 8 --hidden_size 512 --dropout 0 \
		--test_conform_epoch 50 --conform_eval_epoch 50 --generate_eval_epoch 50 --sampling_steps 100 --val_time 0.6 --infer_time 0.9 --infer_noise 0.9;
exit
}
