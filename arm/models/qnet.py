import torch
import torch.nn as nn
import logging
from functools import partial 
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from arm.network_utils import \
    Conv3DInceptionBlock, Conv3DInceptionBlockUpsampleBlock, \
    DenseBlock, SpatialSoftmax3D, \
    Conv3DBlock, Conv3DUpsampleBlock, \
    Conv3DResNetBlock, Conv3DResNetUpsampleBlock
from einops import rearrange, reduce, repeat, parse_shape

class QattentionEmbedNet(nn.Module):
    """ use this to generate embedding feature maps that match q-attention intermediate outputs """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_dense: int,
                 voxel_size: int, 
                 kernels: int,
                 norm: str = None,
                 activation: str = 'relu',
                 dense_feats: int = 32,
                 dev_cfgs: dict = {}
                 ):
        super(QattentionEmbedNet, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._norm = norm
        self._activation = activation
        self._kernels = kernels  
        self._voxel_size = voxel_size
        self._dense_feats = dense_feats
        self._out_dense = out_dense 
        self._dev_cfgs  = dev_cfgs
        block_class = Conv3DInceptionBlock
        upsample_block_class = Conv3DInceptionBlockUpsampleBlock
        if self._dev_cfgs.get('conv3d', False):
            block_class = partial(Conv3DBlock, 
                 kernel_sizes=3, strides=1)
             # Conv3DResNetBlock 
            upsample_block_class =  partial(Conv3DUpsampleBlock,
                kernel_sizes=3, strides=1)
         
         
        spatial_size = self._voxel_size
        self._input_preprocess = block_class(
            self._in_channels, self._kernels, norm=self._norm,
            activation=self._activation)

        d0_ins = self._input_preprocess.out_channels
        print('building d0_in num channels:', d0_ins)
 
        self._down0 = block_class(
            d0_ins, self._kernels, norm=self._norm,
            activation=self._activation, residual=False)
        self._ss0 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down0.out_channels)
        spatial_size //= 2

        d1_ins = self._down0.out_channels
        self._down1 = block_class(
            d1_ins, self._kernels * 2, norm=self._norm,
            activation=self._activation, residual=False)
        self._ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down1.out_channels)
        spatial_size //= 2

        flat_size = self._down0.out_channels * 4 + self._down1.out_channels * 4
        
        k1 = self._down1.out_channels
        if self._voxel_size > 8:
            k1 += self._kernels
            d2_ins = self._down1.out_channels
             
            self._down2 = block_class(
                d2_ins, self._kernels * 4, norm=self._norm,
                activation=self._activation,  residual=False)
            flat_size += self._down2.out_channels * 4
            self._ss2 = SpatialSoftmax3D(
                spatial_size, spatial_size, spatial_size,
                self._down2.out_channels)
            spatial_size //= 2
            
            k2 = self._down2.out_channels
            if self._voxel_size > 16:
                k2 *= 2
                self._down3 = block_class(
                    self._down2.out_channels, self._kernels, norm=self._norm,
                    activation=self._activation, residual=False)
                flat_size += self._down3.out_channels * 4
                self._ss3 = SpatialSoftmax3D(
                    spatial_size, spatial_size, spatial_size,
                    self._down3.out_channels)
                self._up3 = upsample_block_class(
                    self._kernels, self._kernels, scale_factor=2, norm=self._norm,
                    activation=self._activation, residual=False)
            self._up2 = upsample_block_class(
                k2, self._kernels, scale_factor=2, norm=self._norm,
                activation=self._activation, residual=False)
   
        self._up1 = upsample_block_class(
            k1, self._kernels, scale_factor=2, norm=self._norm,
            activation=self._activation, residual=False)

        self._global_maxp = nn.AdaptiveMaxPool3d(1)
        self._local_maxp = nn.MaxPool3d(3, 2, padding=1)

        final_ins = self._kernels * 2
        
        self._final = Conv3DBlock(
            final_ins, self._kernels, kernel_sizes=3,
            strides=1, norm=self._norm, activation=self._activation)
        self._final2 = Conv3DBlock(
            self._kernels, self._out_channels, kernel_sizes=3,
            strides=1, norm=None, activation=None)

        self._ss_final = SpatialSoftmax3D(
            self._voxel_size, self._voxel_size, self._voxel_size,
            self._kernels)
        flat_size += self._kernels * 4
 
        
        self._dense0 = DenseBlock(
            flat_size, self._dense_feats, None, self._activation)
        self._dense1 = DenseBlock(
            self._dense_feats, self._dense_feats, None, self._activation)
        self._dense2 = DenseBlock(
            self._dense_feats, self._out_dense, None, None)

    def forward(self, ins):
        b, _, d, h, w = ins.shape # ins here is rearranged demo of shape b k n ch h w -> (b k) ch n h w'
        x = self._input_preprocess(ins)

        d0 = self._down0(x)
        print('forward Qnet: d0 shape', d0.shape) #[b, 128] 
        ss0 = self._ss0(d0)
        maxp0 = self._global_maxp(d0).view(b, -1)
        down1_in = self._local_maxp(d0)

        d1 = u = self._down1(down1_in)
        ss1 = self._ss1(d1)
        maxp1 = self._global_maxp(d1).view(b, -1)
        print('forward Qnet: d1 shape', d1.shape) #[b, 128] 
        feats = [ss0, maxp0, ss1, maxp1]
        if self._voxel_size > 8:
            down2_in = self._local_maxp(d1)
            # print('forward Qnet: down2_in shape', down2_in.shape) # [b, 128, 4, 4, 4] 
            d2 = u = self._down2(down2_in)
            print('forward Qnet: down2_out shape', d2.shape) # [b, 128, 4, 4, 4]
            feats.extend([self._ss2(d2), self._global_maxp(d2).view(b, -1)])
            if self._voxel_size > 16:
                d3 = self._down3(self._local_maxp(d2))
                feats.extend([self._ss3(d3), self._global_maxp(d3).view(b, -1)])
                u3 = self._up3(d3)
                u = torch.cat([d2, u3], dim=1) 
            
            up2_in = u
            # print('forward Qnet: up2_in shape', up2_in.shape) # [b, 256, 4, 4, 4]
            u2 = self._up2(up2_in)
            print('forward Qnet: up2_out shape', u2.shape) # torch.Size([b, 64, 8, 8, 8])
            u = torch.cat([d1, u2], dim=1)

        up1_in = u
        u1 = self._up1(up1_in)

        f1_in = torch.cat([d0, u1], dim=1)
        # print('forward Qnet: f1 shape', f1_in.shape) #[b, 64, 16, 16, 16]
        f1 = self._final(f1_in) 
        f2 = self._final2(f1) 

        feats.extend([self._ss_final(f1), self._global_maxp(f1).view(b, -1)])
        self.latent_dict = {
            'emb_d0': d0,
            'emb_d1': d1,
            'emb_u1': u1, 
        }
        if self._voxel_size > 8:
            self.latent_dict.update({
                'emb_d2': d2,
                'emb_u2': u2,
            })
        if self._voxel_size > 16:
            self.latent_dict.update({ 
                'emb_d3': d3,
                'emb_u3': u3,
            })
        dense0_in = torch.cat(feats, 1)
        # print('forward Qnet: dense0_in shape', dense0_in.shape) [b, 2048]
        dense0 = self._dense0(dense0_in)
        dense1 = self._dense1(dense0)
        embedding = self._dense2(dense1)

        self.latent_dict.update({
            'dense0': dense0,
            'dense1': dense1, 
            'embedding': embedding,
        })
 
        return embeddings 

if __name__ == '__main__':
    qnet = QattentionEmbedNet(
        in_channels=3,
        out_channels=2,
        voxel_size=16,
        out_dense=32,
        kernels=64,
        norm=None, 
        dense_feats=128,
        activation='lrelu', 
        dev_cfgs={'conv3d':False},
        )
    demo_in = torch.ones(1, 3, 2, 128, 128)
    emb = qnet(demo_in)
    print(emb.shape)
    for k, v in qnet.latent_dict.items():
        print(k, v.shape)