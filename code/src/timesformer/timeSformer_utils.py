import warnings
import torch.onnx
import torch
import logging
import os
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

from src.timesformer.vit_configs import get_b16_config

CONFIGS = {
    # "sup_vitb8": configs.get_b16_config(),
    "sup_vitb16_224": get_b16_config(),

}
vit_cfg = CONFIGS["sup_vitb16_224"]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 101, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


# 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth'
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def to_2tuple(x): return tuple((x, x))


_logger = logging.getLogger(__name__)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained(setting_config, model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, img_size=224,
                    num_frames=8,
                    num_patches=196, attention_type='divided_space_time', pretrained_model="", strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return
    if len(pretrained_model) == 0:
        state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    else:
        try:
            state_dict = load_state_dict(pretrained_model)['model']
        except:
            state_dict = load_state_dict(pretrained_model)

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            _logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != state_dict[classifier_name + '.weight'].size(0):
        # print('Removing the last fully connected layer due to dimensions mismatch ('+str(num_classes)+ ' != '+str(state_dict[classifier_name + '.weight'].size(0))+').', flush=True)
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False
    ## Resizing the positional embeddings in case they don't match
    if num_patches + 1 != state_dict['pos_embed'].size(1):
        print("执行了")
        print("pos_embed.shape", state_dict['pos_embed'].shape)
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed
        print("pos_embed.shape", state_dict['pos_embed'].shape)
    print("pos_embed.shape", state_dict['pos_embed'].shape)

    ## Resizing time embeddings in case they don't match
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'].size(1):
        print("time_embed.shape", state_dict['time_embed'].shape)
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=(num_frames), mode='nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)
        print("time_embed.shape", state_dict['time_embed'].shape)

    ## Initializing temporal attention
    if attention_type == 'divided_space_time':
        new_state_dict = state_dict.copy()
        for key in state_dict:
            if 'blocks' in key and 'attn' in key:
                new_key = key.replace('attn', 'temporal_attn')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
            if 'blocks' in key and 'norm1' in key:
                new_key = key.replace('norm1', 'temporal_norm1')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
        state_dict = new_state_dict

    msg = model.load_state_dict(state_dict, strict=False)
    # for name, param in model.named_parameters():
    #     # param.requires_grad_(True)
    #     if "blocks" in name:
    #         if 'blocks.11' in name:
    #             param.requires_grad_(True)
    #         else:
    #             param.requires_grad_(False)
    #     if "Adapter" in name:
    #         param.requires_grad_(True)
    #     if "cls_token" in name:
    #         param.requires_grad = False
    #     if "time_embed" in name:
    #         param.requires_grad_(False)
    #     if "patch_embed" in name:
    #         param.requires_grad_(False)
    #     if "head" in name:
    #         param.requires_grad_(True)
    #     if name == "pos_embed":
    #         param.requires_grad_(False)
    #     if name == "norm.weight":
    #         param.requires_grad_(True)
    #     if name == "norm.bias":
    #         param.requires_grad_(True)
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    for _, p in model.norm.named_parameters():
        p.requires_grad = True

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert (attention_type in ['divided_space_time', 'space_only', 'joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        elif self.attention_type == 'divided_space_time':
            ## Temporal
            # x: torch.Size([2, 1649, 768])
            xt = x[:, 1:, :]  # torch.Size([2, 1648, 768])
            # xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:, 1:, :] + res_temporal

            ## Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
            cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=112, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape  # B:2 C:3 T:8 H,W:224
        x = rearrange(x, 'b c t h w -> (b t) c h w')  # 要合并F和B成一个维度，有(B,C,T,H,W)->((B,T),C,H,W)
        # [16,3,224,224]
        x = self.proj(x)  # ((b t), dim, h//p, w//p)
        # [16, 768, 14,14]
        W = x.size(-1)  # W:14
        x = x.flatten(2).transpose(1, 2)  # ((b, t), )
        return x, T, W  # x : [(b, t), h//p * w//p, dims]-->[(B,T), nums_patches, dim]


def _conv_filter(state_dict, patch_size=16):
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
