from src.timesformer.timeSformer_utils import trunc_normal_, load_pretrained, default_cfgs
from src.timesformer.timeSformer_utils import PatchEmbed, Block, _conv_filter, to_2tuple
from src.timesformer.PromptBlock import PromptBlock
from src.timesformer.PromptAIMblock import AIMPromptBlock
from torch.nn import functional as F
from torch.nn import Dropout
from functools import partial, reduce
from einops import rearrange
from operator import mul
from torch import nn
import torch
import math


class PromptVisionTransformer(nn.Module):
    def __init__(self, config, prompt_config, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames=8,
                 attention_type='divided_space_time', dropout=0.):
        super().__init__()
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        self.cfg = config
        self.img_size = img_size
        self.patch_size = patch_size
        self.attention_type = attention_type
        self.num_patches = int((self.img_size // self.patch_size) * (self.img_size // self.patch_size))
        self.prompt_config = prompt_config
        self.depth = depth
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if 'k400' not in self.cfg.DATA.NAME:
            self.cls_token.requires_grad = True

        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)
        self.adapter = self.cfg.TRICKS.ADAPTER
        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        if self.adapter:
            self.blocks = nn.ModuleList([
                AIMPromptBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    attention_type=self.attention_type, num_tadapter=1, scale=1)
                for i in range(self.depth)])
        else:
            self.blocks = nn.ModuleList([
                PromptBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    attention_type=self.attention_type)
                for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        ## Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        ## prompt
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.prompt_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.prompt_pos_embed.requires_grad = True

        if self.prompt_config.PROJECT > -1:
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, self.embed_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = self.embed_dim
            self.prompt_proj = nn.Identity()
        # 如果不使用attribute 则随机初始化
        if not self.cfg.PROMPT.ATTRIBUTE:
            if self.prompt_config.INITIATION == "random":
                val = math.sqrt(6. / float(3 * reduce(mul, to_2tuple(self.patch_size), 1) + prompt_dim))
                self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, prompt_dim))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                if self.prompt_config.DEEP:
                    total_d_layer = depth - 1
                    self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, self.num_tokens, prompt_dim))
                    nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            else:
                raise ValueError("Other initiation scheme is not supported")

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def incorporate_prompt(self, x, p):
        B = x.shape[0]
        if self.cfg.PROMPT.ATTRIBUTE:
            x = torch.cat((x[:, :1, :], p.repeat(B // (p.shape[0]), 1, 1), x[:, 1:, :]), dim=1)
        else:
            x = torch.cat((x[:, :1, :],
                           self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                           x[:, 1:, :]), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x

    def forward_deep_prompt(self, embedding_output, p, T, W):
        hidden_states = None
        B = embedding_output.shape[0]  # 2
        num_layers = self.depth  # 12
        # 如果使用attribute
        if self.cfg.PROMPT.ATTRIBUTE:
            for i in range(num_layers):
                if i == 0:
                    hidden_states = self.blocks[i](embedding_output, B, T, W)
                else:
                    # if i <= self.cfg.PROMPT.ATTRIBUTE_DEPTH_NUM - 1:
                    # if i <= 11:
                    if i in self.cfg.PROMPT.ATTRIBUTE_DEPTH_NUM:
                        # print("执行第{}".format(i))
                        hidden_states = torch.cat(
                            (hidden_states[:, :1, :], p.repeat(B // (p.shape[0]), 1, 1),
                             hidden_states[:, (1 + self.num_tokens):, :]), dim=1)
                    hidden_states = self.blocks[i](hidden_states, B, T, W)

            encoded = self.norm(hidden_states)
        else:
            for i in range(num_layers):
                if i == 0:
                    hidden_states = self.blocks[i](embedding_output, B, T, W)
                else:
                    if i <= self.deep_prompt_embeddings.shape[0]:
                        deep_prompt_emb = self.prompt_dropout(
                            self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))
                        hidden_states = torch.cat(
                            (hidden_states[:, :1, :], deep_prompt_emb, hidden_states[:, (1 + self.num_tokens):, :]),
                            dim=1)
                    hidden_states = self.blocks[i](hidden_states, B, T, W)
            encoded = self.norm(hidden_states)
        return encoded

    def forward_features(self, x, p):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        x = self.incorporate_prompt(x, p)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        if x.size(1) != torch.cat((self.prompt_pos_embed, self.pos_embed), dim=1).size(1):  # 重采样pos_embed
            pos_embed = torch.cat((self.prompt_pos_embed, self.pos_embed), dim=1)
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            # pos_emd顺序有问题
            x = x + torch.cat((self.prompt_pos_embed, self.pos_embed), dim=1)
        # dropout
        x = self.pos_drop(x)
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            if T != self.time_embed.size(1):
                # 重采样time_embed
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            # dropout
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)
        return x, T, W

    def forward(self, x, prompt_attribute):
        p = None
        # prompt_attribute:[batch_size, token_num, hidden_dim]
        if self.cfg.PROMPT.ATTRIBUTE:
            p = prompt_attribute
        x, T, W = self.forward_features(x, p)
        if self.prompt_config.DEEP:
            x = self.forward_deep_prompt(x, p, T, W)

        x = x[:, 0]

        x = self.head(x)
        return x


class PromptTimeSformer(nn.Module):
    def __init__(self, config, prompt_cfg, img_size=112, patch_size=16, num_classes=400, num_frames=8,
                 attention_type='divided_space_time',
                 pretrained_model='', **kwargs):
        super(PromptTimeSformer, self).__init__()
        self.pretrained = True
        self.prompt_cfg = prompt_cfg
        self.cfg = config
        self.model = PromptVisionTransformer(config=self.cfg, prompt_config=self.prompt_cfg,
                                             img_size=img_size, num_classes=num_classes,
                                             patch_size=patch_size, embed_dim=768,
                                             depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type,
                                             **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch' + str(patch_size) + '_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        if self.pretrained:
            load_pretrained(self.cfg, self.model, num_classes=self.model.num_classes,
                            in_chans=kwargs.get('in_chans', 3),
                            filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames,
                            num_patches=self.num_patches, attention_type=self.attention_type,
                            pretrained_model=pretrained_model)

    def forward(self, x, prompt_attribute):
        # print(x.shape)
        x = self.model(x, prompt_attribute)
        return x
