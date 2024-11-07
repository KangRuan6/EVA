import torch
import src.utils.logging as logging
from src.models.vit_models import ViT
import os

logger = logging.get_logger("visual_prompt")


def build_model(cfg):
    """
    build model here
    """
    # print('torch.cuda.device_count()',torch.cuda.device_count())
    # assert (cfg.NUM_GPUS < torch.cuda.device_count()), "Cannot use more GPU devices than available"
    # 创建模型--> vit
    model = ViT(cfg)

    log_model_info(model, verbose=cfg.DBG)
    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")
    logger.info("fine tune parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info("name:{},shape:{}".format(name, param.shape))
    return model, device


def build_model_ssv2(cfg):
    model = ViT(cfg)
    log_model_info(model, verbose=cfg.DBG)
    # logger.info(f"Device used for model: {device}")
    # logger.info("fine tune parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info("name:{},shape:{}".format(name, param.shape))
    return model


def log_model_info(model, verbose=False):
    """Logs model info"""
    # 写入模型的参数信息
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f" % (model_grad_params / model_total_params * 100))


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    print("cur_device", cur_device)
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        # 重点！！！
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            # torch.distributed.init_process_group('nccl', init_method='file:///home/.../my_file', world_size=1, rank=0)
            # Make model replica operate on the current device
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '5778'
            # torch.cuda.set_device('cuda:0,1,2,3')
            # torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=4)
            # print("执行开始")
            # torch.distributed.init_process_group(backend='nccl',rank=0,world_size=2)
            # device = torch.device('cuda:1,2')
            # model = torch.nn.parallel.DistributedDataParallel(
            #     module=model, device_ids=cur_device, output_device=cur_device,
            #     find_unused_parameters=True,
            # )
            # print("执行结束")
            # torch.distributed.init_process_group(backend='nccl', rank=0, world_size=4)
            torch.distributed.init_process_group(backend='nccl', rank=0, world_size=4)
            model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    else:
        model = model.to(cur_device)
    return model, cur_device
