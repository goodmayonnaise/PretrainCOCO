# Copyright (c) OpenMMLab. All rights reserved.
import math

from abc import ABCMeta
from copy import deepcopy
from collections import OrderedDict
from typing import Union, Optional, Tuple, Sequence, List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import build_norm_layer, build_conv_layer
from mmcv.cnn.bricks.transformer import build_dropout, FFN

from mmdet.utils import OptConfigType

from mmengine.utils import to_2tuple
from mmengine.model import ModuleList, BaseModule
from mmengine.model.weight_init import trunc_normal_, trunc_normal_init, constant_init
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.logging import MMLogger

from mmyolo.registry import MODELS

from ..utils import make_divisible

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

INF = 1e8


def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, input_size=224, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = input_size
        self.patch_size = patch_size
        self.H, self.W = input_size[0] // stride, input_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=1)
                            #   padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
    

class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmengine.ConfigDict`, optional): The Config for
            initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 conv_type: str = 'Conv2d',
                 kernel_size: int = 16,
                 stride: int = 16,
                 padding: Union[int, tuple, str] = 'corner',
                 dilation: int = 1,
                 bias: bool = True,
                 norm_cfg: OptConfigType = None,
                 input_size: Union[int, tuple] = None,
                 init_cfg: OptConfigType = None) -> None:
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int]]:
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                # embed_dims=embed_dims,
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class DynamicGate(nn.Module):
    def __init__(self, in_channels):
        super(DynamicGate, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Global Average Pooling
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.fc(x)
        return x * gate  # 입력 feature에 따라 동적으로 강조/억제

# 사용 예제
fusion_feature = torch.randn(1, 256, 16, 16)  # 예제 feature
gate_module = DynamicGate(256)
output = gate_module(fusion_feature)


@MODELS.register_module()
class SwinTFusionNeck(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 depths,
                 num_heads, 
                 window_size,
                 mlp_ratio,
                 qkv_bias, 
                 drop_rate,
                 attn_drop_rate,
                 drop_path_rate,
                 out_indices,
                 with_cp,
                 convert_weights,
                 widen_factor,
                 featmap_strides,
                 patch_norm=True,
                 qk_scale=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 patch_size=4,
                 frozen_stages=False,
                 fusion_ver=1): #fusion version 
        
        super().__init__()

        self.in_channels = in_channels
        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self.embed_dim = embed_dim
        self.featmap_strides = featmap_strides
        self.widen_factor = widen_factor
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.with_cp = with_cp
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        self.fusion_ver = fusion_ver

        self.patch_embed = PatchEmbed(
            in_channels=in_channels[0],
            embed_dims=embed_dim,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size//2,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)
 
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.eps = 1e-6
            
        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        self.stage1 = SwinBlockSequence(
                embed_dims=embed_dim, # in channel , out channel 
                num_heads=num_heads[0],
                feedforward_channels=mlp_ratio * embed_dim,
                depth=depths[0],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                downsample=None,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None
                )
        
        self.stage1 = SwinBlockSequence(
                embed_dims=embed_dim, 
                num_heads=num_heads[0],
                feedforward_channels=mlp_ratio * embed_dim,
                depth=depths[0],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                downsample=None,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None
                )
        
        self.downsample1 = OverlapPatchEmbed(
            input_size=1024//self.featmap_strides[1],
            patch_size=4,
            stride=2, 
            in_channels=embed_dim,
            embed_dim=int(self.embed_dim*2*self.widen_factor)
        )

        self.stage2 = SwinBlockSequence(
            embed_dims=int(self.embed_dim*2*self.widen_factor),
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * int(self.embed_dim*2*self.widen_factor),
            depth=depths[1],
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
            downsample=None,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            init_cfg=None
        )
        self.downsample2 = OverlapPatchEmbed(
            input_size=1024//self.featmap_strides[1],
            patch_size=4,
            stride=2, 
            in_channels=int(self.embed_dim*2*self.widen_factor),
            embed_dim=int(self.embed_dim*4*self.widen_factor)
        )

        self.stage3 = SwinBlockSequence(
            embed_dims=int(self.embed_dim*4*self.widen_factor),
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * int(self.embed_dim*4*self.widen_factor),
            depth=depths[2],
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
            downsample=None,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            init_cfg=None
        )
        if self.fusion_ver == 1: # fusion selection
           self.gate1 = nn.Conv2d(self.embed_dim*2, self.embed_dim, 1)
           self.gate2 = nn.Conv2d(int(self.embed_dim*4*self.widen_factor), int(self.embed_dim*2*self.widen_factor), 1)
        elif self.fusion_ver == 2:
            self.conv1x1_1 = nn.Conv2d(self.embed_dim*2, self.embed_dim, 1) # feature 정리 
            self.gate1 = nn.Conv2d(self.embed_dim, self.embed_dim, 1) # fusion selection 
            self.conv1x1_2 = nn.Conv2d(int(self.embed_dim*4*self.widen_factor), int(self.embed_dim*2*self.widen_factor), 1)
            self.gate2 = nn.Conv2d(int(self.embed_dim*2*self.widen_factor), int(self.embed_dim*2*self.widen_factor), 1)

        for i in out_indices :
            if i == 1:
                layer = build_norm_layer(norm_cfg, int(self.embed_dim))[1]
            elif i == 2:
                layer = build_norm_layer(norm_cfg, int(self.embed_dim*2*self.widen_factor))[1]
            elif i == 3:
                layer = build_norm_layer(norm_cfg, int(self.embed_dim*4*self.widen_factor))[1]
            layer_name = f"norm{i}"
            self.add_module(layer_name, layer)


    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False


    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            # if self.use_abs_pos_embed:
            #     self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    
    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            # if self.use_abs_pos_embed:
            #     trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)
    
    
    def forward(self,  inputs: List[torch.Tensor]) -> tuple:
        p3, p4, p5 = inputs # 192 128 128
        x, hw_shape = self.patch_embed(p3) # 4096(64*64) 384
        x = self.drop_after_pos(x)
        
        # stage 1
        for block in self.stage1.blocks:
            x = block(x, hw_shape)
        norm_layer1 = getattr(self, 'norm1')
        s1 = norm_layer1(x)
        s1 = s1.view(-1, *hw_shape, self.embed_dim).permute(0, 3, 1, 2).contiguous() # 384 64 64

        # stage 2
        if self.fusion_ver == 1:
            gate1 = torch.sigmoid(self.gate1(torch.cat([s1, p4], dim=1))) # fusion 
            fusion1 = gate1*s1 + (1-gate1)*p4        
        elif self.fusion_ver == 2:
            fusion1 = self.conv1x1_1(torch.cat([s1, p4], 1))
            gate1 = torch.sigmoid(self.gate1(fusion1))
            fusion1 = gate1*s1 + (1-gate1)*p4        
        s2, h, w = self.downsample1(fusion1)
        s2 = self.drop_after_pos(s2)
        for block in self.stage2.blocks:
            s2 = block(s2, (h, w))
        norm_layer2 = getattr(self, "norm2")
        s2 = norm_layer2(s2)
        s2 = s2.view(-1, *(h,w), int(self.embed_dim*2*self.widen_factor)).permute(0, 3, 1, 2).contiguous()
        
        # stage 3
        if self.fusion_ver == 1:
            gate2 = torch.sigmoid(self.gate2(torch.cat([s2, p5], 1))) # fusion 
            fusion2 = gate2*s2 + (1-gate2)*p5
        elif self.fusion_ver == 2:
            fusion2 = self.conv1x1_2(torch.cat([s2, p5], 1))
            gate2 = torch.sigmoid(self.gate2(fusion2))
            fusion2 = gate2*s2 + (1-gate2)*p5
        s3, h, w = self.downsample2(fusion2)
        for block in self.stage3.blocks:
            s3 = block(s3, (h, w))
        norm_layer3 = getattr(self, "norm3")
        s3 = norm_layer3(s3)
        s3 = s3.view(-1, *(h, w), int(self.embed_dim*4*self.widen_factor)).permute(0, 3, 1, 2).contiguous()
        
        return [s1, s2, s3]
    
        
    
    # def forward(self,  inputs: List[torch.Tensor]) -> tuple:
    #     p3, p4, p5 = inputs # 192 128 128
    #     x, hw_shape = self.patch_embed(p3) # 4096(64*64) 384
    #     x = self.drop_after_pos(x)
        
    #     # stage 1
    #     for block in self.stage1.blocks:
    #         x = block(x, hw_shape)
    #     norm_layer1 = getattr(self, 'norm1')
    #     s1 = norm_layer1(x)
    #     s1 = s1.view(-1, *hw_shape, self.embed_dim).permute(0, 3, 1, 2).contiguous() # 384 64 64

    #     # stage 2
    #     gate1 = torch.sigmoid(self.gate1(torch.cat([s1, p4], dim=1))) # fusion 
    #     fusion1 = gate1*s1 + (1-gate1)*p4        
    #     s2, h, w = self.downsample1(fusion1)
    #     s2 = self.drop_after_pos(s2)
    #     for block in self.stage2.blocks:
    #         s2 = block(s2, (h, w))
    #     norm_layer2 = getattr(self, "norm2")
    #     s2 = norm_layer2(s2)
    #     s2 = s2.view(-1, *(h,w), int(self.embed_dim*2*self.widen_factor)).permute(0, 3, 1, 2).contiguous()
        
    #     # stage 3
    #     gate2 = torch.sigmoid(self.gate2(torch.cat([s2, p5], 1))) # fusion 
    #     fusion2 = gate2*s2 + (1-gate2)*p5
    #     s3, h, w = self.downsample2(fusion2)
    #     for block in self.stage3.blocks:
    #         s3 = block(s3, (h, w))
    #     norm_layer3 = getattr(self, "norm3")
    #     s3 = norm_layer3(s3)
    #     s3 = s3.view(-1, *(h, w), int(self.embed_dim*4*self.widen_factor)).permute(0, 3, 1, 2).contiguous()
        
    #     return [s1, s2, s3]

