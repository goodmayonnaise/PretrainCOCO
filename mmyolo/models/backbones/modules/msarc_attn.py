import torch
import torch.nn as nn
from torch import Tensor
from .adaptive_rotated_conv import AdaptiveRotatedConv2d
from .routing_function import RountingFunction
from typing import List, Optional, Sequence, Tuple, Union
from mmengine.model import BaseModule
from mmengine.config import ConfigDict

ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, arc, conv2_stride=1, dilation=1, padding=0, kernel_number=4):
        super().__init__()
        if arc:
            self.conv = AdaptiveRotatedConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size, 
                    stride=conv2_stride,
                    padding=padding,
                    groups=1,
                    dilation=dilation,
                    rounting_func=RountingFunction(
                        in_channels=in_channels,
                        kernel_number=kernel_number
                    ))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=conv2_stride,
                                    padding=padding,
                                    groups=1,
                                    dilation=dilation)
            
        self.norm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, input):
        return self.act(self.norm(self.conv(input)))

class MSARCModule(nn.Module):
    def __init__(self,
                 dim: int,
                 dilation: Sequence[int] = (1, 1, 1),
                 padding: Sequence[int] = (1, 2, 3),
                 kernel_number: int = 4,
                 conv_num:int = 3,
                 ):
        super().__init__()
        self.num_conv = conv_num
        self.ops = nn.ModuleDict()
        mid_dim_lst = [dim // conv_num] * conv_num
        
        if sum(mid_dim_lst) != dim:
            for i in range(dim % conv_num):
                mid_dim_lst[i] += 1
        self.middle_dim = mid_dim_lst
        
        for i in range(self.num_conv):
            self.ops.update({'ARCBottleneck{}'.format(i):ConvModule(
                in_channels=self.middle_dim[i],
                out_channels=self.middle_dim[i],
                kernel_size=3,
                arc=True,
                padding=padding[i],
                conv2_stride=1,
                dilation=dilation[i],
                kernel_number=kernel_number
            )})
        self.fusion = ConvModule(
            dim,
            dim,
            kernel_size=1,
            arc=False,
        )
    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        short = x
        if not isinstance(x, list):
            _,c,_,_ = x.shape
            x = torch.chunk(x, self.num_conv, dim=1)
            
        feats = []
        for i in range(len(self.ops)):
            feats.append(self.ops['ARCBottleneck{}'.format(i)](x[i]))
        
        feats = torch.cat(feats, dim=1)
        return self.fusion(feats) + short

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output
    
class CSPLayerWithMSARCAtten(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = False,  # shortcut
            kernel_number:int = 4,
            dilation:Sequence[int] = (1,2,3),
            padding:Sequence[int] = (1,2,3),
            conv_number:int = 3,
            spattn:bool = True,
            chattn:bool = True,
            init_cfg: OptMultiConfig = None) -> None:
        
        super().__init__(init_cfg=init_cfg)
        self.spattn = spattn
        self.chattn = chattn
        self.add_identity = add_identity
        self.expand_ratio = 0.5
        self.num_block = num_blocks
        
        self.main_conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            arc=False,
            padding=0,
            kernel_number=kernel_number)
        
        expand_ratio_arc = list(pow(expand_ratio, i+1) for i in range(num_blocks))
        self.mid_channels = [int(out_channels * ratio) for ratio in expand_ratio_arc]
        
        self.blocks = nn.ModuleList(
            MSARCModule(
                dim=self.mid_channels[i],
                dilation=dilation,
                kernel_number=kernel_number,
                conv_num=conv_number,
                padding=padding
                ) for i in range(num_blocks))
        
        
        if self.chattn:
            self.ca = ChannelAttention(out_channels)
        
        if self.spattn:
            self.sa = SpatialAttention()
            
        self.final_conv = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            arc=False,
            padding=0,
            )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        if self.add_identity:
            identity = x
            
        out_list = []
        _,c,_,_ = x.shape
        x_main = self.main_conv(x)
        feat = list(x_main.split(int(c*self.expand_ratio), 1))
        out_list.append(feat[0])
        learning_feats = feat[1]
        
        for i, block in enumerate(self.blocks):
            learning_feats = block(learning_feats) 
            m_c= int(learning_feats.size(1)*self.expand_ratio)
            split_feat, learning_feats = learning_feats.split(m_c, 1)
            out_list.append(split_feat)
            out_list.append(learning_feats) if i == self.num_block-1 else None

        feature = torch.cat(out_list, dim=1)
        
        if self.chattn:
            ca = self.ca(feature)
            attn_feat = feature*ca
            
        if self.spattn:
            sa = self.sa(attn_feat)
            attn_feat = attn_feat*sa
        
        if self.add_identity:
            return self.final_conv(attn_feat) + identity
        else:
            return self.final_conv(attn_feat)