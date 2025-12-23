# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import re
import subprocess

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from util import datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from msm_mae import models_mae

from msm_mae.engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('output_dir', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int) # original default=800
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_freq', default=20, type=int)
    parser.add_argument('--eval_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16x16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default='80x208', type=str,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--no_cls_token', action='store_true',
                        help='Do not use [CLS] token if set.')
    parser.set_defaults(no_cls_token=False)

    parser.add_argument('--dec_pos_2d', action='store_true',
                        help='Use a 2-D positional embeddings on decoder. It is a 1-D by default.')
    parser.set_defaults(dec_pos_2d=False)

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 128')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR') # original default=40

    # Dataset parameters
    parser.add_argument('--data_path', default='audio_lms', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='trainingfiles', type=str,
                        help='dataset definition')
    parser.add_argument('--norm_stats', default='None', type=str,
                        help='dataset normalization stats')

    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


import matplotlib.pyplot as plt


def visualize_reconstruction(args, device, model, save_path):
    ds, files = datasets.build_viz_dataset(args)
    if len(ds) == 0:
        print(f'(Skipped visualization which require samples in {args.data_path}/vis_samples folder.)')
        return
    batch = torch.stack([ds[i] for i in range(len(ds))])
    model.eval()
    with torch.no_grad():
        _, recons, _, masks = model.forward_viz(batch.to(device))
    save_path.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(files):
        # as .npy
        np.save(f"{save_path}/recon_{Path(file).name}", recons[i].cpu().numpy())
        # as .png
        fig = plt.figure(figsize=[12, 8 if batch[0].shape[-1] < 310 else 6])
        for j, img in enumerate([batch[i][0], recons[i][0], masks[i]]):
            ax = fig.add_subplot(3, 1, j + 1)
            ax.imshow(img.cpu().numpy(), origin='lower')
        plt.margins(x=0, y=0)
        fig.savefig(f'{save_path}/recon_{Path(file).stem}.png', bbox_inches = 'tight')


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = datasets.build_dataset(args)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    eff_batch_size = args.batch_size * args.accum_iter * torch.cuda.device_count()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=eff_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](img_size=args.input_size, norm_pix_loss=args.norm_pix_loss,
        use_cls_token=(not args.no_cls_token), use_2d_dec_pos_embd=args.dec_pos_2d)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 128

    print("base lr: %.2e" % (args.lr * 128 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    model = torch.nn.DataParallel(model).to(device)
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch1 = epoch + 1
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch1 % args.save_freq == 0 or epoch1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch1)
            # visualize reconstructions
            out_dir = Path(args.output_dir)/str(epoch)
            visualize_reconstruction(args, device, model_without_ddp, out_dir)
            # run the external evaluator
            if epoch1 % args.eval_freq == 0 or epoch1 == args.epochs:
                abspath = Path(f'{args.output_dir}/checkpoint-{epoch1}.pth').absolute()
                subprocess.Popen(['/bin/bash', './quick_eval.sh', abspath])

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.log_dir:
        args.log_dir = args.output_dir
    args.input_size = [int(x) for x in args.input_size.split('x')]
    args.patch_size = [int(x) for x in re.match(r'.+\_patch([0-9]+x[0-9]+)$', str(args.model)).group(1).split('x')]
    args.norm_stats = eval(args.norm_stats) if args.norm_stats else None
    print(args)
    main(args)
