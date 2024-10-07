{
export CUDA_VISIBLE_DEVICES='0,1,2,3';

python train_lm_conf.py --dataset "QM9-df" --root "./data/tordf_qm9_v2" --batch_size 256 --num_workers 4 --max_epochs 2000 --mode train --diff_version dgt --save_every_n_epochs 50  --filename 'dgt' --llm_model "acharkq/MoLlama"  --test_conform_epoch 50 --conform_eval_epoch 50 --generate_eval_epoch 50  --sampling_steps 100;
exit
}