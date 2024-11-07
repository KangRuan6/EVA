from einops import rearrange
from src.timesformer.timeSformer_utils import Attention, DropPath, Mlp
from torch import nn
import torch.onnx
import torch


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


class AIMPromptBlock(nn.Module):

    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time',
                 num_tadapter=2, scale=1):
        super().__init__()
        self.attention_type = attention_type
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)
        # Adapter
        self.MLP_Adapter = Adapter(dim, skip_connect=False)
        self.S_Adapter = Adapter(dim)
        self.scale = scale
        self.T_Adapter = Adapter(dim, skip_connect=False)
        self.num_tadapter = num_tadapter
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(dim)

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        ## temporal adaptation
        xt = x[:, 1:, :]
        xt = rearrange(xt, 'b (hw t) m -> (b hw) t m', b=B, t=T)
        if self.num_tadapter == 2:
            res_temporal = self.drop_path(
                self.T_Adapter(self.temporal_attn(self.T_Adapter_in(self.temporal_norm1(xt)))))
        else:
            res_temporal = self.drop_path(self.T_Adapter(self.temporal_attn(self.temporal_norm1(xt))))
        res_temporal = rearrange(res_temporal, '(b hw) t m -> b (hw t) m', b=B, t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x[:, 1:, :] + res_temporal

        ## spatial adaptation
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)

        xs = xt
        xs = rearrange(xs, 'b (hw t) m -> (b t) hw m', b=B, t=T)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.drop_path(self.S_Adapter(self.attn(self.norm1(xs))))

        ### Taking care of CLS token
        cls_token = res_spatial[:, 0, :]
        cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
        cls_token = torch.mean(cls_token, 1, True)
        res_spatial = res_spatial[:, 1:, :]
        res_spatial = rearrange(res_spatial, '(b t) hw m -> b (hw t) m', b=B, t=T)
        res = res_spatial
        x = xt + res

        ## Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)

        xn = self.norm2(x)
        x_mlp = self.mlp(xn)
        x_mlp_adapter = self.drop_path(self.scale * self.MLP_Adapter(xn))

        x = x + x_mlp + x_mlp_adapter

        return x


if __name__ == '__main__':
    aim = AIMPromptBlock()
    x = torch.rand(16, 1649, 768)
    y = aim(x, 16, 8, 14)
    print(x.shape)
