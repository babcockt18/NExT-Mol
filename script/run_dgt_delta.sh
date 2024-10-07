{
filename="dgt_d01_delta";
ckpt_path="all_checkpoints/${filename}/last.ckpt";
if [ ! -d "$ckpt_path" ]; then
        ckpt_path="None"
fi

python train_lm_conf.py --dataset "QM9-df" --root "./data/tordf_qm9_v2" --batch_size 256 --num_workers 4 --max_epochs 500 --mode train --save_every_n_epochs 50 --ckpt_path ${ckpt_path} --filename $filename  --llm_model "acharkq/MoLlama" --test_conform_epoch 50 --conform_eval_epoch 50 --check_val_every_n_epoch 5  --generate_eval_epoch 50 --sampling_steps 100 --use_llm --diff_ckpt "all_checkpoints/dgt_not_com_d01/epoch=1999.ckpt/converted.ckpt" --delta_train;
exit
}