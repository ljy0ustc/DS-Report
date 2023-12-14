import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model.model_interface import MInterface
from data.data_interface import DInterface
from utils import load_model_path_by_args
import torch


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_f1',
        mode='max',
        patience=100,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_f1',
        dirpath=args.ckpt_dir,
        filename='{epoch:02d}-{val_f1:.3f}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        every_n_epochs=10
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("load checkpoints from {}".format(args.ckpt_path))

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    # args.callbacks = load_callbacks()
    # args.logger = logger

    trainer = Trainer.from_argparse_args(args)
    if args.mode == 'train':
        trainer.fit(model=model, datamodule=data_module)
    else:
        trainer.test(model=model, datamodule=data_module)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--devices', default=-1, type=int)
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--check_val_every_n_epoch', default=5, type=int)
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', default='step', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=1, type=int)
    parser.add_argument('--lr_decay_rate', default=0.97, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-6, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='comp_data', type=str)
    parser.add_argument('--data_dir', default='data/ref', type=str)
    parser.add_argument('--model_name', default='comp_net', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
    # Model Hyperparameters
    parser.add_argument('--img_dim', default=768, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--class_num', default=2, type=int)
    parser.add_argument('--layer_num', default=2, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    #parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=50)

    args = parser.parse_args()


    main(args)
