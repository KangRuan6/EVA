import os

os.environ["WANDB_API_KEP"] = "aa47517e5628d9fffa3725bd1e6c02093158539e"
from src.utils.launch import default_argument_parser, logging_train_setup
from src.utils.setup_utils import setup

args = default_argument_parser().parse_args()
cfg = setup(args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPUS_id)

import torch
import warnings
import numpy as np
import random
import src.utils.logging as logging
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model, build_model_ssv2
from dataloaders.dataset import VideoDataset
from dataloaders.HMDB_dataset import get_hmdb51_loaders
from dataloaders.video_dataloader import get_video_dataloader
from dataloaders.k400_tiny_dataloader import K400_tiny_dataloader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from TimeSformer.dataloader_k400_tiny import load_k400
from dataloaders.ssv2_dataloader import get_dataloader

if cfg.TEST.WANDB:
    import wandb

    wandb.login()

warnings.filterwarnings("ignore")


def get_data_loaders(cfg, logger, split_mode, dataset_name, clip_len):
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    logger.info("Loading {} data".format(split_mode))
    dataset = VideoDataset(cfg=cfg, dataset=dataset_name, split=split_mode, clip_len=clip_len)
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    data_loader = DataLoader(dataset,
                             shuffle=(False if sampler else True),
                             sampler=sampler,
                             num_workers=16,
                             pin_memory=cfg.DATA.PIN_MEMORY,
                             drop_last=drop_last,
                             batch_size=cfg.DATA.BATCH_SIZE)

    return data_loader


def get_loaders(cfg, logger):
    dataset_name = cfg.DATA.NAME
    train_loader = get_data_loaders(cfg, logger, split_mode='train',
                                    dataset_name=dataset_name, clip_len=16)
    val_loader = get_data_loaders(cfg, logger, split_mode='val',
                                  dataset_name=dataset_name, clip_len=16)
    test_loader = get_data_loaders(cfg, logger, split_mode='test',
                                   dataset_name=dataset_name, clip_len=16)
    return train_loader, val_loader, test_loader


def train(cfg, args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    ########### 加载数据 ###########
    logger.info("loading dataset...")
    if cfg.DATA.NAME == 'ucf101':
        train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    if cfg.DATA.NAME == 'hmdb51':
        # train_loader, val_loader, test_loader = get_hmdb51_loaders(cfg, logger)
        train_loader, val_loader, test_loader = get_video_dataloader(cfg )
    if cfg.DATA.NAME == 'k400-tiny':
        # train_loader, val_loader, test_loader = K400_tiny_dataloader(cfg)
        yaml_name = './EVA/code/TimeSformer/configs/Kinetics/TimeSformer_spaceOnly_8x32_224.yaml'
        train_loader, val_loader, test_loader = load_k400(yaml_name)
    if cfg.DATA.NAME == 'SSV2':
        train_loader, val_loader, test_loader = get_dataloader(cfg, batach_size=cfg.DATA.BATCH_SIZE,
                                                               num_worker=16,
                                                               num_segments=16)
    logger.info("Loading dataset finished...")

    ########### 创建模型 ###########
    logger.info("Constructing models...")
    if cfg.DATA.NAME == 'SSV2':
        model = build_model_ssv2(cfg)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[0, 1, 2, 3])
    else:
        model, cur_device = build_model(cfg)

    # model = torch.nn.parallel.DistributedDataParallel(
    #     module=model, device_ids=[cur_device], output_device=cur_device,
    #     find_unused_parameters=True,
    # )
    # from torch.nn import DataParallel
    # torch.cuda.set_device(1)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
    # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1])
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=['1','2'])
    # model_without_ddp = model.module
    ########### 加载评估 ###########
    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()

    ########### 加载训练 ###########
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    if cfg.SOLVER.TOTAL_EPOCH == 0:
        trainer.eval_classifier(test_loader, "test", 0)


if __name__ == '__main__':

    if cfg.TEST.WANDB:
        wandb.init(
            project="VAPT",
            config={
                "learning_rate": cfg.SOLVER.BASE_LR,
                "architecture": cfg.MODEL.TYPE,
                "dataset": cfg.DATA.NAME,
                "epochs": cfg.SOLVER.TOTAL_EPOCH,
            },
            name='%s_%s_%s_%s_%s_%s_%s' % (cfg.DATA.NAME, cfg.SOLVER.BASE_LR, cfg.GPUS_id,
                                           cfg.SOLVER.TOTAL_EPOCH, cfg.SOLVER.OPTIMIZER,
                                           cfg.SOLVER.WARMUP_EPOCH,
                                           cfg.MODEL.INTRO),
        )

    train(cfg, args)

    if cfg.TEST.WANDB:
        wandb.finish()
