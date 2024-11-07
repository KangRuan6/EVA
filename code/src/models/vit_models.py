import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
from src.models.mlp import MLP
from src.utils import logging
from src.timesformer.PromptTimesformer import PromptTimeSformer
from src.timesformer.promptmae import vit_base_patch16_224, vit_small_patch16_224,vit_huge_patch16_224
from src.timesformer.i3dpt import I3D_model
import numpy as np
import os

logger = logging.get_logger("visual_prompt")


def build_vit_sup_models(cfg=None, prompt_cfg=None, num_class=101, load_pretrain=True):
    model = None
    if cfg.MODEL.TYPE == "vit":
        if load_pretrain:
            model = PromptTimeSformer(cfg, prompt_cfg, num_classes=num_class, img_size=cfg.DATA.CROPSIZE,
                                      attention_type='divided_space_time',
                                      pretrained_model='./EVA/code/src/timesformer/TimeSformer_divST_8x32_224_K400.pyth')

        else:
            model = PromptedTimeSformer(cfg, prompt_cfg, num_classes=num_class, img_size=cfg.DATA.CROPSIZE)
    if cfg.MODEL.TYPE == 'videoMAE':
        model = vit_base_patch16_224(config=cfg, prompt_config=prompt_cfg, num_classes=num_class,
                                     img_size=cfg.DATA.CROPSIZE, embed_dim=768,
                                     pretrained=cfg.MODEL.CHECK)
        logger.info(cfg.MODEL.CHECK)
                                 

    if cfg.MODEL.TYPE == 'CNN':
        model = I3D_model(num_classes=num_class,
                          check_point="./EVA/code/src/timesformer/i3d_model_rgb.pth")
    if cfg.MODEL.TYPE == 'videoMAE-h':
        model = vit_huge_patch16_224(config=cfg, prompt_config=prompt_cfg, num_classes=num_class,
                                     img_size=cfg.DATA.CROPSIZE, embed_dim=1280,
                                     pretrained=cfg.MODEL.CHECK)
        logger.info(cfg.MODEL.CHECK)
    return model, num_class


class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            self.prompt_cfg = cfg.MODEL.PROMPT
        else:
            self.prompt_cfg = None

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False

        if cfg.MODEL.TRANSFER_TYPE == "adapter":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None

        self.build_backbone(self.prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        self.setup_side()
        # if 'k400' not in self.cfg.DATA.NAME:
        #     self.setup_head(cfg)

    def setup_side(self):
        if self.cfg.MODEL.TRANSFER_TYPE != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        ############## 创建模型 ###########
        self.enc, self.feat_dim = build_vit_sup_models(cfg=cfg,
                                                       prompt_cfg=self.prompt_cfg,
                                                       num_class=cfg.DATA.NUMBER_CLASSES,
                                                       load_pretrain=load_pretrain)

    #     if transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
    #         for k, p in self.enc.named_parameters():
    #             if "prompt" not in k and "embeddings.patch_embeddings.weight" not in k and "embeddings.patch_embeddings.bias" not in k:
    #                 p.requires_grad = False
    #
    #     # elif transfer_type == "prompt":
    #     #     for k, p in self.enc.named_parameters():
    #     #         if "prompt" not in k:
    #     #             p.requires_grad = False
    #
    #     elif transfer_type == "prompt+bias":
    #         for k, p in self.enc.named_parameters():
    #             if "prompt" not in k and 'bias' not in k:
    #                 p.requires_grad = False
    #
    #     elif transfer_type == "prompt-noupdate":
    #         for k, p in self.enc.named_parameters():
    #             p.requires_grad = False
    #
    #     elif transfer_type == "cls":
    #         for k, p in self.enc.named_parameters():
    #             if "cls_token" not in k:
    #                 p.requires_grad = False
    #
    #     elif transfer_type == "cls-reinit":
    #         nn.init.normal_(
    #             self.enc.transformer.embeddings.cls_token,
    #             std=1e-6
    #         )
    #
    #         for k, p in self.enc.named_parameters():
    #             if "cls_token" not in k:
    #                 p.requires_grad = False
    #
    #     elif transfer_type == "cls+prompt":
    #         for k, p in self.enc.named_parameters():
    #             if "prompt" not in k and "cls_token" not in k:
    #                 p.requires_grad = False
    #
    #     elif transfer_type == "cls-reinit+prompt":
    #         nn.init.normal_(
    #             self.enc.transformer.embeddings.cls_token,
    #             std=1e-6
    #         )
    #         for k, p in self.enc.named_parameters():
    #             if "prompt" not in k and "cls_token" not in k:
    #                 p.requires_grad = False
    #
    #     # adapter
    #     elif transfer_type == "adapter":
    #         for k, p in self.enc.named_parameters():
    #             if "adapter" not in k:
    #                 p.requires_grad = False
    #
    #     elif transfer_type == "end2end":
    #         logger.info("Enable all parameters update during training")
    #
    #     # else:
    #     #     raise ValueError("transfer type {} is not supported".format(
    #     #         transfer_type))
    #
    # # def setup_head(self, cfg):
    # #     self.head = MLP(
    # #         input_dim=self.feat_dim,
    # #         mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
    # #                  [cfg.DATA.NUMBER_CLASSES],  # noqa
    # #         special_bias=True
    # #     )

    def forward(self, x, prompt_attribute, return_feature=False):
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x, prompt_attribute)  # batch_size x self.feat_dim

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        if return_feature:
            return x, x
        # if 'k400' not in self.cfg.DATA.NAME:
        #     x = self.head(x)

        return x

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x
