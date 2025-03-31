# Copyright (c) OpenMMLab. All rights reserved.
import copy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import List, Sequence, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.dist import get_dist_info
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess, make_divisible
from .yolov8_head import YOLOv8Head


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

INF = 1e8


class PixelShuffleDecode(nn.Module):
    def __init__(self, 
                 in_channels,
                 feat_type='p3'
                 ):
        super(PixelShuffleDecode, self).__init__()
        self.in_channels = in_channels
        self.feat_type = feat_type
        self.init_layers()
        
    def init_layers(self):

        
        if self.feat_type == 'p3':      
            self.decode1 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels*4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode2 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels*4, 3, 1, 1),
                nn.PixelShuffle(2)
            )  
            self.decode3 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels*2, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode4 = nn.Sequential(
                nn.Conv2d(self.in_channels//2, self.in_channels, 3, 1, 1),
                nn.PixelShuffle(2)
            )
        
        elif self.feat_type == 'p4':
            self.decode1 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels*4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode2 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels*2, 3, 1, 1),
                nn.PixelShuffle(2)
            )  
            self.decode3 = nn.Sequential(
                nn.Conv2d(self.in_channels//2, self.in_channels, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode4 = nn.Sequential(
                nn.Conv2d(self.in_channels//4, self.in_channels//2, 3, 1, 1),
                nn.PixelShuffle(2)
            )
        
        elif self.feat_type == 'p5':
            self.decode1 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels*2, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode2 = nn.Sequential(
                nn.Conv2d(self.in_channels//2, self.in_channels, 3, 1, 1),
                nn.PixelShuffle(2)
            )  
            self.decode3 = nn.Sequential(
                nn.Conv2d(self.in_channels//4, self.in_channels//2, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.decode4 = nn.Sequential(
                nn.Conv2d(self.in_channels//8, self.in_channels//4, 3, 1, 1),
                nn.PixelShuffle(2)
            )

    def forward(self, x):
        global_feats, local_feats = x
        
        up1 = self.decode1(local_feats)
        up2 = self.decode2(up1)
        up3 = self.decode3(up2)
        local_feats = self.decode4(up3)
        
        return global_feats, local_feats


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class CustomViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.rearrange = Rearrange('b (ph pw) d -> b d ph pw', ph=image_height//patch_height, pw=image_width//patch_width)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img) # b (h/p w/p) embed dim
        b, n, _ = x.shape # h/p*w/p = n

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) 
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] 
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        cls_tokens, patchs = x[:, 0], x[:, 1:]
        
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        global_feats = self.to_latent(cls_tokens)
        global_feats = self.mlp_head(global_feats) # sigmoid는?
        
        local_feats = self.rearrange(patchs)
        
        return global_feats, local_feats
        # return local_feats


@MODELS.register_module()
class ViTHeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 vit_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max
        self.vit_cfg = vit_cfg

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            # cls_pred[-1].bias.data[:self.num_classes] = math.log(
            #     5 / self.num_classes / (1024 / stride)**2)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""

        self.vit_layers = nn.ModuleList()
        self.vit_decoder_layers = nn.ModuleList()

        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.cls_preds.append(nn.Conv2d(in_channels=self.in_channels[0]*3//4, out_channels=self.num_classes, kernel_size=1))
        self.cls_preds.append(nn.Conv2d(in_channels=self.in_channels[1]*3//8, out_channels=self.num_classes, kernel_size=1))
        self.cls_preds.append(nn.Conv2d(in_channels=self.in_channels[2]*3//16, out_channels=self.num_classes, kernel_size=1))

        self.reg_preds = nn.ModuleList()
        self.reg_preds.append(
            nn.Sequential(
                    nn.Conv2d(self.in_channels[0]*3//4, out_channels=self.in_channels[0]*3//2, kernel_size=1),
                    nn.Conv2d(self.in_channels[0]*3//2, out_channels=4 * self.reg_max, kernel_size=1)
                )
        )
        self.reg_preds.append(
            nn.Sequential(
                    nn.Conv2d(self.in_channels[1]*3//8, out_channels=self.in_channels[1]*3//4, kernel_size=1),
                    nn.Conv2d(self.in_channels[1]*3//4, out_channels=4 * self.reg_max, kernel_size=1)
                )
        )
        self.reg_preds.append(
            nn.Sequential(
                    nn.Conv2d(self.in_channels[2]*3//16, out_channels=self.in_channels[2]*3//8, kernel_size=1),
                    nn.Conv2d(self.in_channels[2]*3//8, out_channels=4 * self.reg_max, kernel_size=1)
                )
        )

        for i in range(self.num_levels):
            self.vit_layers.append(
                    CustomViT(
                        image_size=1024//self.featmap_strides[i],
                        patch_size=self.vit_cfg.patch_size,
                        num_classes=self.num_classes,
                        dim=self.in_channels[i]*3,
                        depth=self.vit_cfg.depth,
                        heads=self.vit_cfg.num_heads,
                        channels=self.in_channels[i],
                        mlp_dim=self.vit_cfg.mlp_dim)
            )
            self.vit_decoder_layers.append(
                    PixelShuffleDecode(in_channels=self.in_channels[i]*3,
                                       feat_type=f"p{i+3}"),
            )

        self.cls_scores_g = nn.Linear(self.num_classes*3, self.num_classes)
        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels
        # return multi_apply(self.forward_single, x, self.cls_preds,
        #                    self.reg_preds)
        preds = multi_apply(self.forward_single, x, self.vit_layers, self.vit_decoder_layers,
                             self.cls_preds, self.reg_preds)
        if self.training:
            cls_scores_g, cls_scores_l, reg_preds, reg_dist_preds = preds
        else:
            cls_scores_g, cls_scores_l, reg_preds = preds 
        
        cls_scores_g = torch.cat(cls_scores_g, 1)
        cls_scores_g = self.cls_scores_g(cls_scores_g)

        if self.training:
            return cls_scores_g, cls_scores_l, reg_preds, reg_dist_preds
        else: 
            return cls_scores_g, cls_scores_l, reg_preds

    def forward_single(self, x: torch.Tensor, vit_layer:nn.ModuleList, vit_decoder_layer:nn.ModuleList,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        
        vit_enc = vit_layer(x)
        cls_logit_g, local_feats = vit_decoder_layer(vit_enc)
         
        cls_logit_l = cls_pred(local_feats)
        bbox_dist_preds = reg_pred(local_feats)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit_g, cls_logit_l, bbox_preds, bbox_dist_preds
        else:
            return cls_logit_g, cls_logit_l, bbox_preds



@MODELS.register_module()
class ViTHead(YOLOv8Head):
    """YOLOv8Head head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 vit_config=ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 loss_ce=None):
        self.vit_config=vit_config
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_ce = MODELS.build(loss_ce)
        # YOLOv8 doesn't need loss_obj
        self.loss_obj = None

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def loss_by_feat(
            self,
            cls_scores_g: Sequence[Tensor],
            cls_scores_l: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores_l
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores_l[0].dtype,
                device=cls_scores_l[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores_l
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        
        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        cls_scores_g = cls_scores_g.sigmoid()
        cls_scores_g = cls_scores_g.flatten()
        cls_scores_g_sum = cls_scores_g.sum().clamp(min=1)
        
        # gt_labels_g = torch.cat([F.one_hot(i.unique().long(), self.num_classes).sum(0) for i in gt_labels]) # origin
        gt_labels_g = torch.cat([F.one_hot(i.unique().long(), self.num_classes).sum(0).detach().cpu() for i in gt_labels]).to(gt_labels.device).detach()
        # gt_labels_g = torch.cat([F.one_hot(i.unique().long(), self.num_classes).sum(0).detach().cpu() for i in gt_labels]).to(gt_labels.device)
        # gt_labels_g = torch.cat([F.one_hot(i.unique().long(), self.num_classes).sum(0) for i in gt_labels]).detach()
        loss_ce = self.loss_ce(cls_scores_g, gt_labels_g)
        loss_ce /= cls_scores_g_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()


        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_ce=loss_ce * num_imgs * world_size, 
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size)
    
    def predict_by_feat(self,
                        cls_scores_g: List[Tensor],
                        cls_scores_l: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores_l) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores_l) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores_l]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores_l[0].dtype,
                device=cls_scores_l[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores_l
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
