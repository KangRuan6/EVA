import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from collections import OrderedDict
from easydict import EasyDict
import torch
from dataloaders.datasets.video_datasets import build_dataset
from dataloaders.datasets.kinetics import build_training_dataset


def get_args_parser():
    parser = argparse.ArgumentParser('AdaptFormer fine-tuning for action recognition', add_help=False)

    # Experiment parameters
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default="/home/yqx/yqx_softlink/data/HMDB51/hmdb51", type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=174, type=int, help='number of the classification types')

    # video data parameters
    parser.add_argument('--data_set', default='HMDB51', choices=['SSV2', 'HMDB51'], type=str, help='dataset')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_sample', type=int, default=1, help='Repeated_aug (default: 1)')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=4)
    parser.add_argument('--test_num_crop', type=int, default=3)
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')

    # efficient GPU parameters
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--inception', default=False, action='store_true', help='whether use INCPETION mean and std')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--config-file', default='config')
    return parser


def get_video_dataloader(cfg):
    args = get_args_parser()
    args = args.parse_args()
    args.batch_size = cfg.DATA.BATCH_SIZE
    if args.data_set == 'SSV2':
        args.nb_classes = 174
    elif args.data_set == 'HMDB51':
        args.nb_classes = 51
    else:
        raise ValueError(args.data_set)
    dataset_train = build_training_dataset(args)
    ## 这里为了迎合adaptformer的写法，故意反过来写了

    dataset_test, _ = build_dataset(is_train=False, test_mode=False, args=args,cfg=cfg)
    dataset_val, _ = build_dataset(is_train=False, test_mode=True, args=args,cfg=cfg)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    # for i_batch, sample in enumerate(data_loader_train):
    #     print(i_batch, sample['image'].size(), sample['label'].size())
    return data_loader_train, data_loader_val, data_loader_test
