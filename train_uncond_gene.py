import argparse
import torch
import pickle
import warnings
import lightning as L
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import CSVLogger
from model.uncond_gen_pl import UncondGenPL
import os
from rdkit import RDLogger
from data_provider.geom_drugs_jodo_dm import GeomDrugsJODODFDM
from data_provider.qm9_jodo_dm import QM9DM

# Suppress PyTorch deprecation warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, message="'has_cuda' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_cudnn' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_mps' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_mkldnn' is deprecated")

disable_compile = torch.cuda.get_device_name(0).find('AMD') >= 0

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['DEEPSPEED_TIMEOUT'] = '180' # set NCCL timeout to 3 hours instead of 30 minutes to avoid broken pipe error
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
RDLogger.DisableLog('rdApp.*')
# warnings.filterwarnings(action='ignore')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)



def main(args):
    current_epoch = 0
    current_step = 0
    seed = args.seed
    if args.ckpt_path and args.ckpt_path != 'None':
        path = os.path.join(args.ckpt_path, 'checkpoint/mp_rank_00_model_states.pt')
        ckpt = torch.load(path, map_location='cpu')
        current_step = ckpt['global_step']
        seed += current_step
    elif args.init_checkpoint:
        assert (not args.ckpt_path) or args.ckpt_path == 'None'
        ckpt = torch.load(args.init_checkpoint, map_location=torch.device('cpu'))
        current_epoch = ckpt['epoch']
        seed += current_epoch
    L.seed_everything(seed)

    # tokenizer
    tokenizer = UncondGenPL.init_tokenizer(args)
    print('before inserting total tokens:', len(tokenizer))
    if args.dataset == 'QM9-jodo':
        dm = QM9DM(args.root, args.num_workers, args.batch_size, tokenizer, args)
    elif args.dataset == 'Geom-drugs-jodo':
        dm = GeomDrugsJODODFDM(args.root, args.num_workers, args.batch_size, tokenizer, args)
    else:
        raise NotImplementedError(f"dataset {args.dataset} not implemented")
    print('after inserting total tokens:', len(tokenizer))

    # model
    if args.init_checkpoint:
        model = UncondGenPL(args, tokenizer=tokenizer, max_sf_tokens=dm.max_sf_tokens, noise_scheduler=dm.noise_scheduler, pos_std=dm.pos_std)
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=torch.device('cpu'))['state_dict'], strict=True)
        print(f"Loading model from {args.init_checkpoint}")
    else:
        model = UncondGenPL(args, tokenizer=tokenizer, max_sf_tokens=dm.max_sf_tokens, noise_scheduler=dm.noise_scheduler, pos_std=dm.pos_std)
    model.train()

    print('total params:', sum(p.numel() for p in model.parameters()))

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/",
                                         filename='{epoch:02d}',
                                         every_n_epochs=args.save_every_n_epochs,
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))

    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/",
                                         filename='last{epoch:02d}',
                                         mode='min',
                                         every_n_epochs=args.cache_epoch,
                                         save_top_k=1,
                                         save_on_train_epoch_end=True,
                                         save_last='link',))

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("multi-gpu training")
        if args.strategy_name == 'deepspeed':
            strategy = L.pytorch.strategies.DeepSpeedStrategy(stage=2)
        else:
            strategy = L.pytorch.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        print("single-gpu training")
        strategy = 'auto'
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    ## this is to recompile the code when switching different training methods
    torch._dynamo.reset()

    if args.mode.find('eval') >= 0:
        model.diffusion_model = torch.compile(model.diffusion_model, dynamic=True, fullgraph=False, disable=disable_compile)

    if device_count > 1:
        print("multi-gpu training")
        if args.strategy_name == 'deepspeed':
            strategy = L.pytorch.strategies.DeepSpeedStrategy(stage=2)
        else:
            strategy = L.pytorch.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        print("single-gpu training")
        strategy = 'auto'

    print(f"============Train: {args.max_epochs} epochs============")
    trainer = L.Trainer(
        num_nodes=args.num_nodes,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        detect_anomaly=args.detect_anomaly,
        # num_sanity_val_steps=0,
        # limit_train_batches=10,
        # limit_val_batches=10,
    )
    if args.mode in ['train']:
        if args.ckpt_path and args.ckpt_path != 'None':
            print('resume training from epoch ', current_epoch)
            trainer.fit_loop.epoch_progress.current.completed = current_epoch
            trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
        else:
            trainer.fit(model, datamodule=dm)
        trainer.validate(model, datamodule=dm)
    elif args.mode in {'eval', 'eval_gen', 'eval_conf', 'eval_1d_gen', 'eval_test_conform'}:
        if args.ckpt_path and args.ckpt_path != 'None':
            print('resume training from epoch ', current_epoch)
            trainer.validate(model, datamodule=dm, ckpt_path=args.ckpt_path)
        else:
            trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='auto')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--delta_max_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--delta_train', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser = UncondGenPL.add_model_specific_args(parser)
    parser.add_argument('--dataset', type=str, default='QM9-df')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--cache_epoch', type=int, default=5)
    parser.add_argument('--load_test_only', action='store_true', default=False)
    # parser.add_argument('--eval_smiles_path', type=str, default=None)
    parser = QM9DM.add_model_specific_args(parser)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

