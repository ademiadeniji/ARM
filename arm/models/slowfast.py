""" Use SpatialTemporal ResNet (i.e. Slow from SlowFast paper) network as video encoder """

import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from arm.models.helpers import * 
from omegaconf import OmegaConf

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {
    18: (2, 2, 2, 2),
    50: (3, 4, 6, 3), 
    101: (3, 4, 23, 3)
    }

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}

class TempResNet(nn.Module):
    """
    credit: https://github1s.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py 
    
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg): 
        super(TempResNet, self).__init__()
        self.norm_module = get_norm(cfg) 
        self.num_pathways = 1
        self._construct_network(cfg)
        init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        self._device = None 

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model with specified temporal kernel dimensions.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = ResNetBasicStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM[0],
            dim_out=width_per_group,  # same 64 across Res18 to Res152
            kernel=temp_kernel[0][0] + [7, 7],
            stride=[1, 2, 2],
            padding=[temp_kernel[0][0][0] // 2, 3, 3],
            norm_module=self.norm_module,
        )

        self.s2 = ResStage(
            dim_in=[width_per_group],  
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )
        # NOTE(Mandi): removed the detection option here
        self.head = ResNetBasicHead(
            dim_in=[width_per_group * 32], # default to 2048
            dim_out=cfg.MODEL.OUT_DIM,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES // pool_size[0][0],
                    cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = x[None] # NOTE(modified here to fit the pathway convention)
        
        x = self.s2(x) 
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])

        x = self.s3(x) 
        x = self.s4(x) 
        x = self.s5(x) 
        x = self.head(x)
        return x

    def get_device(self):
        return self._device 
    
    def set_device(self, device):
        self._device = device 
        self.to(device)

    def debug_forward(self, x, bboxes=None):
        x = self.s1(x)
        x = x[None] # NOTE(modified here to fit the pathway convention)
        print(x.shape)
        x = self.s2(x)
        print(x[0].shape)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
            print(x[0].shape)
        x = self.s3(x)
        print(x[0].shape)
        x = self.s4(x)
        print(x[0].shape)
        x = self.s5(x)
        print(x[0].shape)
        x = self.head(x)
        return x


if __name__ == '__main__':
    # test run a model
    model_cfg = OmegaConf.load(\
        '/home/mandi/ARM/conf/encoders/SlowRes.yaml')
    slow_18 = TempResNet(model_cfg)
    inp = torch.ones((1, 3, 2, 128, 128))
    print(slow_18.debug_forward(inp).shape)