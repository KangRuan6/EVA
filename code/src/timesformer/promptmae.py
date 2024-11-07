# --------------------------------------------------------
# References:
# VideoMAE: https://github.com/MCG-NJU/VideoMAE
# timm: https://github.com/rwightman/pytorch-image-models
# --------------------------------------------------------
from easydict import EasyDict
from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import os
from src.timesformer.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed

from src.utils import logging

logger = logging.get_logger("visual_prompt")


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,
            config=None, cache_key: str = None, layer_id=0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        # unbind q, k, v linear project
        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)  # q = self.q_proj(x)
        _k = F.linear(input=x, weight=self.k_proj.weight, bias=None)  # k = self.k_proj(x
        k = self._shape(_k, N, B).view(B * self.num_heads, -1, self.head_dim)
        _v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
        v = self._shape(_v, N, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        ################################
        q = q * self.scale  # fix: q scaling before prefix concat
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, config=None, layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            config=config, layer_id=layer_id,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # rewrite FFN here
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)
        #
        # if True:
        #     self.adaptmlp = Adapter(self.config, dropout=drop, bottleneck=config.ffn_num,
        #                             init_option=config.ffn_adapter_init_option,
        #                             adapter_scalar=config.ffn_adapter_scalar,
        #                             adapter_layernorm_option=config.ffn_adapter_layernorm_option
        #                             )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
        # adapt_x = self.adaptmlp(x, add_residual=False)

        residual = x
        x = self.act(self.fc1(self.norm2(x)))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        # if self.config.ffn_adapt:
        #     if self.config.ffn_option == 'sequential':
        #         x = self.adaptmlp(x)
        # elif self.config.ffn_option == 'parallel':
        # x = x + adapt_x
        # else:
        #     raise ValueError(self.config.ffn_adapt)
        x = residual + x
        return x


class Adapter(nn.Module):
    def __init__(self, d_features=768, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(d_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(d_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, d_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class AIMBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, config=None, layer_id=None, num_tadapter=2, scale=1):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            config=config, layer_id=layer_id,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        # rewrite FFN here
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        # Adapter
        self.MLP_Adapter = Adapter(dim, skip_connect=False)
        self.S_Adapter = Adapter(dim)
        self.scale = scale
        self.T_Adapter = Adapter(dim, skip_connect=False)
        self.num_tadapter = num_tadapter
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(dim)

    def forward(self, x):
        if self.num_tadapter == 2:
            x = x + self.drop_path(self.T_Adapter(self.attn(self.T_Adapter_in(self.norm1(x)))))
        else:
            x = x + self.drop_path(self.T_Adapter(self.attn(self.norm1(x))))
        x = x + self.S_Adapter(self.attn(self.norm1(x)))
        xn = self.norm2(x)
        x_mlp = self.fc2(self.act(self.fc1(xn)))
        x_mlp_adapter = self.drop_path(self.scale * self.MLP_Adapter(xn))
        x = x + x_mlp + x_mlp_adapter
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
                num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, init_scale=0., all_frames=16,
                 tubelet_size=2, use_mean_pooling=True, config=None, prompt_config=None):
        super().__init__()
        self.cfg = config
        self.prompt_config = prompt_config
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.adapter = self.cfg.TRICKS.ADAPTER
        if self.adapter:
            self.blocks = nn.ModuleList([
                AIMBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, config=None, layer_id=i)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, config=None, layer_id=i)
                for i in range(depth)])

        if use_mean_pooling:
            self.fc_norm = norm_layer(embed_dim)
            # self.fc_norm = nn.Identity()
            self.norm = nn.Identity()
        else:
            raise NotImplementedError
            self.norm = norm_layer(embed_dim)
        self.global_pool = use_mean_pooling

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

        # VPT
        self.num_tokens = self.prompt_config.NUM_TOKENS
        if not self.cfg.PROMPT.ATTRIBUTE:
            # properly registered
            self.vpt_prompt = nn.ParameterList(
                [nn.Parameter(torch.empty(1, self.num_tokens, embed_dim)) for _ in range(depth)])
            for eee in self.vpt_prompt:
                torch.nn.init.xavier_uniform_(eee.data)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, p):
        x = self.patch_embed(x)
        B, _, _ = x.size()
        if self.pos_embed is not None:
            # x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device)

        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.cfg.PROMPT.ATTRIBUTE:
                eee = p.expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            else:
                eee = self.vpt_prompt[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x)
            x = x[:, self.num_tokens:, :]

        x = self.norm(x)
        if self.global_pool:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x, p):
        B, _, num_frames, _, _ = x.shape
        x = self.forward_features(x, p)
        x = self.head(x)
        return x


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_224(pretrained=False, embed_dim=768, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrained)
        print("Load pre-trained checkpoint from: %s" % pretrained)
        if 'model' in checkpoint:
            raw_checkpoint_model = checkpoint['model']
        elif 'module' in checkpoint:
            raw_checkpoint_model = checkpoint['module']
        else:
            raw_checkpoint_model = checkpoint

        # TODO: refine
        if os.path.basename(pretrained).startswith('vit'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                print(k)
                if k.startswith('blocks.'):
                    if '.mlp' in k:
                        k = k.replace('.mlp', '')
                        checkpoint_model[k] = v
                    if 'attn.qkv' in k:
                        qi, ki, vi = torch.split(v, v.shape[0] // 3, dim=0)
                        spk = k.split('.')
                        q_flag = '.'.join(spk[0:3] + ['q_proj', 'weight'])
                        k_flag = '.'.join(spk[0:3] + ['k_proj', 'weight'])
                        v_flag = '.'.join(spk[0:3] + ['v_proj', 'weight'])
                        checkpoint_model[q_flag] = qi
                        checkpoint_model[k_flag] = ki
                        checkpoint_model[v_flag] = vi
                    else:
                        checkpoint_model[k] = v  # remove 'encoder.' prefix
                if k.startswith('patch_embed'):
                    checkpoint_model[k] = v
            # del checkpoint_model['norm.weight']
            # del checkpoint_model['norm.bias']
            # del checkpoint_model['fc_norm.weight']
            # del checkpoint_model['fc_norm.bias']

        elif os.path.basename(pretrained).startswith('videomae'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                if k.startswith('encoder.'):
                    checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(pretrained).startswith('self'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                if k.startswith('encoder.'):
                    checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(pretrained).startswith('mae'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                if k.startswith('encoder.'):
                    checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(pretrained).startswith('finetune'):
            checkpoint_model = raw_checkpoint_model
        elif os.path.basename(pretrained) == "vit_base_patch16_224_in21k_tongzhan_new.pth":
            checkpoint_model = raw_checkpoint_model
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(pretrained).startswith('swin_base_patch244'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model['state_dict'].items():
                if k.startswith('backbone.'):
                    checkpoint_model[k[9:]] = v
        else:
            raise ValueError("Warning: Double Check!")

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        logger.info(msg)
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

        for name, p in model.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
        for _, p in model.fc_norm.named_parameters():
            p.requires_grad = True
        return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, embed_dim=768, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=embed_dim, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrained)
        print("Load pre-trained checkpoint from: %s" % pretrained)
        if 'model' in checkpoint:
            raw_checkpoint_model = checkpoint['model']
        elif 'module' in checkpoint:
            raw_checkpoint_model = checkpoint['module']
        else:
            raw_checkpoint_model = checkpoint

        # TODO: refine
        if os.path.basename(pretrained).startswith('vit'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                print(k)
                if k.startswith('blocks.'):
                    if '.mlp' in k:
                        k = k.replace('.mlp', '')
                        checkpoint_model[k] = v
                    if 'attn.qkv' in k:
                        qi, ki, vi = torch.split(v, v.shape[0] // 3, dim=0)
                        spk = k.split('.')
                        q_flag = '.'.join(spk[0:3] + ['q_proj', 'weight'])
                        k_flag = '.'.join(spk[0:3] + ['k_proj', 'weight'])
                        v_flag = '.'.join(spk[0:3] + ['v_proj', 'weight'])
                        checkpoint_model[q_flag] = qi
                        checkpoint_model[k_flag] = ki
                        checkpoint_model[v_flag] = vi
                    else:
                        checkpoint_model[k] = v  # remove 'encoder.' prefix
                if k.startswith('patch_embed'):
                    checkpoint_model[k] = v
            # del checkpoint_model['norm.weight']
            # del checkpoint_model['norm.bias']
            # del checkpoint_model['fc_norm.weight']
            # del checkpoint_model['fc_norm.bias']

        elif os.path.basename(pretrained).startswith('videomae'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                if k.startswith('encoder.'):
                    checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(pretrained).startswith('mae'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                if k.startswith('encoder.'):
                    checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(pretrained).startswith('finetune'):
            checkpoint_model = raw_checkpoint_model
        elif os.path.basename(pretrained) == "vit_base_patch16_224_in21k_tongzhan_new.pth":
            checkpoint_model = raw_checkpoint_model
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(pretrained).startswith('swin_base_patch244'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model['state_dict'].items():
                if k.startswith('backbone.'):
                    checkpoint_model[k[9:]] = v
        else:
            raise ValueError("Warning: Double Check!")

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        logger.info(msg)
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

        for name, p in model.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
        for _, p in model.fc_norm.named_parameters():
            p.requires_grad = True
        return model
