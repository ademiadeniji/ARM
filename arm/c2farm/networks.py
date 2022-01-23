import torch
import torch.nn as nn
import logging
from functools import partial 
import torch.nn.functional as F 
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from arm.network_utils import \
    Conv2DBlock, Conv3DInceptionBlock, Conv3DInceptionBlockUpsampleBlock, \
    DenseBlock, SpatialSoftmax3D, \
    Conv3DBlock, Conv3DUpsampleBlock, \
    Conv3DResNetBlock, Conv3DResNetUpsampleBlock
from typing import List, Union, Tuple
from einops import rearrange, reduce, repeat, parse_shape
LRELU_SLOPE = 0.02 
IMG_SIZE=128 

def sample_gumbel_softmax(context, tau=0.01, eps=1e-20, discrete=True): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  assert len(context.shape) == 2
  _, d = context.shape 
  sampled_unif = torch.rand(context.shape)
  sampled_gumbel = -torch.log(-torch.log(sampled_unif + eps) + eps).to(context.device)
  y = context + sampled_gumbel
  sampled = F.softmax( y / tau, dim=1)
  if discrete:
      y_hard = F.one_hot( torch.argmax(sampled, axis=1), num_classes=d) 
      return (y_hard - sampled).detach() + sampled

  return sampled

class Qattention3DNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_dense: int,
                 voxel_size: int,
                 low_dim_size: int,
                 kernels: int,
                 norm: str = None,
                 activation: str = 'relu',
                 dense_feats: int = 32,
                 include_prev_layer = False,):
        super(Qattention3DNet, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._norm = norm
        self._activation = activation
        self._kernels = kernels
        self._low_dim_size = low_dim_size
        self._build_calls = 0
        self._voxel_size = voxel_size
        self._dense_feats = dense_feats
        self._out_dense = out_dense
        self._include_prev_layer = include_prev_layer

    def build(self):
        use_residual = False
        self._build_calls += 1
        if self._build_calls != 1:
            raise RuntimeError('Build needs to be called once.')

        spatial_size = self._voxel_size
        self._input_preprocess = Conv3DInceptionBlock(
            self._in_channels, self._kernels, norm=self._norm,
            activation=self._activation)

        d0_ins = self._input_preprocess.out_channels
        if self._include_prev_layer:
            PREV_VOXEL_CHANNELS = 0
            self._input_preprocess_prev_layer = Conv3DInceptionBlock(
                self._in_channels + PREV_VOXEL_CHANNELS, self._kernels, norm=self._norm,
                activation=self._activation)
            d0_ins += self._input_preprocess_prev_layer.out_channels

        if self._low_dim_size > 0:
            self._proprio_preprocess = DenseBlock(
                self._low_dim_size, self._kernels, None, self._activation)
            d0_ins += self._kernels

        self._down0 = Conv3DInceptionBlock(
            d0_ins, self._kernels, norm=self._norm,
            activation=self._activation, residual=use_residual)
        self._ss0 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down0.out_channels)
        spatial_size //= 2
        self._down1 = Conv3DInceptionBlock(
            self._down0.out_channels, self._kernels * 2, norm=self._norm,
            activation=self._activation, residual=use_residual)
        self._ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down1.out_channels)
        spatial_size //= 2

        flat_size = self._down0.out_channels * 4 + self._down1.out_channels * 4

        k1 = self._down1.out_channels
        if self._voxel_size > 8:
            k1 += self._kernels
            self._down2 = Conv3DInceptionBlock(
                self._down1.out_channels, self._kernels * 4, norm=self._norm,
                activation=self._activation,  residual=use_residual)
            flat_size += self._down2.out_channels * 4
            self._ss2 = SpatialSoftmax3D(
                spatial_size, spatial_size, spatial_size,
                self._down2.out_channels)
            spatial_size //= 2
            k2 = self._down2.out_channels
            if self._voxel_size > 16:
                k2 *= 2
                self._down3 = Conv3DInceptionBlock(
                    self._down2.out_channels, self._kernels, norm=self._norm,
                    activation=self._activation, residual=use_residual)
                flat_size += self._down3.out_channels * 4
                self._ss3 = SpatialSoftmax3D(
                    spatial_size, spatial_size, spatial_size,
                    self._down3.out_channels)
                self._up3 = Conv3DInceptionBlockUpsampleBlock(
                    self._kernels, self._kernels, 2, norm=self._norm,
                    activation=self._activation, residual=use_residual)
            self._up2 = Conv3DInceptionBlockUpsampleBlock(
                k2, self._kernels, 2, norm=self._norm,
                activation=self._activation, residual=use_residual)

        self._up1 = Conv3DInceptionBlockUpsampleBlock(
            k1, self._kernels, 2, norm=self._norm,
            activation=self._activation, residual=use_residual)

        self._global_maxp = nn.AdaptiveMaxPool3d(1)
        self._local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self._final = Conv3DBlock(
            self._kernels * 2, self._kernels, kernel_sizes=3,
            strides=1, norm=self._norm, activation=self._activation)
        self._final2 = Conv3DBlock(
            self._kernels, self._out_channels, kernel_sizes=3,
            strides=1, norm=None, activation=None)

        self._ss_final = SpatialSoftmax3D(
            self._voxel_size, self._voxel_size, self._voxel_size,
            self._kernels)
        flat_size += self._kernels * 4

        if self._out_dense > 0:
            self._dense0 = DenseBlock(
                flat_size, self._dense_feats, None, self._activation)
            self._dense1 = DenseBlock(
                self._dense_feats, self._dense_feats, None, self._activation)
            self._dense2 = DenseBlock(
                self._dense_feats, self._out_dense, None, None)

    def forward(self, ins, proprio, prev_layer_voxel_grid):
        b, _, d, h, w = ins.shape
        x = self._input_preprocess(ins)

        if self._include_prev_layer:
            y = self._input_preprocess_prev_layer(prev_layer_voxel_grid)
            x = torch.cat([x, y], dim=1)

        if self._low_dim_size > 0:
            p = self._proprio_preprocess(proprio)
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, d, h, w)
            x = torch.cat([x, p], dim=1)

        d0 = self._down0(x)
        ss0 = self._ss0(d0)
        maxp0 = self._global_maxp(d0).view(b, -1)
        d1 = u = self._down1(self._local_maxp(d0))
        ss1 = self._ss1(d1)
        maxp1 = self._global_maxp(d1).view(b, -1)

        feats = [ss0, maxp0, ss1, maxp1]

        if self._voxel_size > 8:
            d2 = u = self._down2(self._local_maxp(d1))
            feats.extend([self._ss2(d2), self._global_maxp(d2).view(b, -1)])
            if self._voxel_size > 16:
                d3 = self._down3(self._local_maxp(d2))
                feats.extend([self._ss3(d3), self._global_maxp(d3).view(b, -1)])
                u3 = self._up3(d3)
                u = torch.cat([d2, u3], dim=1)
            u2 = self._up2(u)
            u = torch.cat([d1, u2], dim=1)

        u1 = self._up1(u)
        f1 = self._final(torch.cat([d0, u1], dim=1))
        trans = self._final2(f1)

        feats.extend([self._ss_final(f1), self._global_maxp(f1).view(b, -1)])

        self.latent_dict = {
            'd0': d0.mean(-1).mean(-1).mean(-1),
            'd1': d1.mean(-1).mean(-1).mean(-1),
            'u1': u1.mean(-1).mean(-1).mean(-1),
            'trans_out': trans,
        }

        rot_and_grip_out = None
        if self._out_dense > 0:
            dense0 = self._dense0(torch.cat(feats, 1))
            dense1 = self._dense1(dense0)
            rot_and_grip_out = self._dense2(dense1)
            self.latent_dict.update({
                'dense0': dense0,
                'dense1': dense1,
                'dense2': rot_and_grip_out,
            })

        if self._voxel_size > 8:
            self.latent_dict.update({
                'd2': d2.mean(-1).mean(-1).mean(-1),
                'u2': u2.mean(-1).mean(-1).mean(-1),
            })
        if self._voxel_size > 16:
            self.latent_dict.update({
                'd3': d3.mean(-1).mean(-1).mean(-1),
                'u3': u3.mean(-1).mean(-1).mean(-1),
            })

        return trans, rot_and_grip_out

class Qattention3DNetWithContext(Qattention3DNet):
    """Use a separate Denseblock to process context
    Q: pass on the processed context or not? """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_dense: int,
                 voxel_size: int,
                 low_dim_size: int,
                 kernels: int,
                 inp_context_size: int,
                 use_context: bool = True, 
                 encode_context: bool = True,
                 encode_context_size: int = 16, 
                 encode_context_hidden: int = -1, 
                 norm: str = None,
                 activation: str = 'relu',
                 dense_feats: int = 32,
                 include_prev_layer = False, # has been false  
                 dev_cfgs: dict = {},   
                 ):
        super(Qattention3DNet, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._norm = norm
        self._activation = activation
        self._kernels = kernels
        self._low_dim_size = low_dim_size
        self._build_calls = 0
        self._voxel_size = voxel_size
        self._dense_feats = dense_feats
        self._out_dense = out_dense
        self._include_prev_layer = include_prev_layer
        # context related
        # self._use_prev_context = use_prev_context 
        self._inp_context_size = inp_context_size
        self._encode_context_size = encode_context_size 
        self._encode_context_hidden = encode_context_hidden
        self._use_context = use_context 
        self._encode_context = encode_context
        self._dev_cfgs = dev_cfgs
        print('Using in-development cfgs:', dev_cfgs)
         
        print(f'Qattention3DNet: Input context embedding size: {inp_context_size}, output size {encode_context_size if encode_context else inp_context_size}')

    def build(self):
        block_class = Conv3DInceptionBlock
        upsample_block_class = Conv3DInceptionBlockUpsampleBlock
        if self._dev_cfgs.get('conv3d', False):
            block_class = partial(Conv3DBlock, 
                 kernel_sizes=3, strides=1)
             # Conv3DResNetBlock 
            upsample_block_class =  partial(Conv3DUpsampleBlock,
                kernel_sizes=3, strides=1)
        
        use_residual = False
        self._build_calls += 1
        if self._build_calls != 1:
            raise RuntimeError('Build needs to be called once.')

        spatial_size = self._voxel_size
        self._input_preprocess = block_class(
            self._in_channels, self._kernels, norm=self._norm,
            activation=self._activation)

        d0_ins = self._input_preprocess.out_channels
        if self._include_prev_layer:
            PREV_VOXEL_CHANNELS = 0
            self._input_preprocess_prev_layer = block_class(
                self._in_channels + PREV_VOXEL_CHANNELS, self._kernels, norm=self._norm,
                activation=self._activation)
            d0_ins += self._input_preprocess_prev_layer.out_channels

        if self._low_dim_size > 0:
            self._proprio_preprocess = DenseBlock(
                self._low_dim_size, self._kernels, None, self._activation)
            d0_ins += self._kernels

        # Note context size can also equal _kernels if this is intermediate net and use_prev_context is True 
        if self._use_context:
            if self._encode_context:
                if self._dev_cfgs.get('ctxt_conv3d', False):
                    self.ctxt_conv3d = nn.Conv3d(
                    in_channels=8192, out_channels=512, 
                    kernel_size=[3,1,1], 
                    stride=2)
                    nn.init.kaiming_uniform_(self.ctxt_conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
                    nn.init.zeros_(self.ctxt_conv3d.bias)
                    self.ctxt_activate = nn.LeakyReLU(negative_slope=LRELU_SLOPE)

                self._context_preprocess = DenseBlock(
                    self._inp_context_size, self._encode_context_size, None, self._activation)
                if self._encode_context_hidden > 0:
                    self._context_preprocess = DenseBlock(
                        self._inp_context_size, self._encode_context_hidden, None, self._activation)
                    self._context_preprocess_2 = DenseBlock(
                        self._encode_context_hidden, self._encode_context_size, None, self._activation)
                d0_ins += self._encode_context_size # exactly the same as how low_dim_size is added
            else:
                logging.info('Warning - Not encoding context in Qattention3DNetWithContext')
                d0_ins += self._inp_context_size 
        self._down0 = block_class(
            d0_ins, self._kernels, norm=self._norm,
            activation=self._activation, residual=use_residual)
        self._ss0 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down0.out_channels)
        spatial_size //= 2

        d1_ins = self._down0.out_channels
        if self._use_context and self._dev_cfgs.get('cat_down1', False):
            d1_ins = d1_ins + self._encode_context_size if self._encode_context else d1_ins + self._inp_context_size 
        self._down1 = block_class(
            d1_ins, self._kernels * 2, norm=self._norm,
            activation=self._activation, residual=use_residual)
        self._ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down1.out_channels)
        spatial_size //= 2

        flat_size = self._down0.out_channels * 4 + self._down1.out_channels * 4
        
        k1 = self._down1.out_channels
        if self._voxel_size > 8:
            k1 += self._kernels
            d2_ins = self._down1.out_channels
            if self._use_context and self._dev_cfgs.get('cat_down2', False):
                d2_ins = d2_ins + self._encode_context_size if self._encode_context else d2_ins + self._inp_context_size 

            self._down2 = block_class(
                d2_ins, self._kernels * 4, norm=self._norm,
                activation=self._activation,  residual=use_residual)
            flat_size += self._down2.out_channels * 4
            self._ss2 = SpatialSoftmax3D(
                spatial_size, spatial_size, spatial_size,
                self._down2.out_channels)
            spatial_size //= 2
            
            k2 = self._down2.out_channels
            if self._use_context and self._dev_cfgs.get('cat_up2', False):
                k2 = k2 + self._encode_context_size if self._encode_context else k2 + self._inp_context_size
            if self._voxel_size > 16:
                k2 *= 2
                self._down3 = block_class(
                    self._down2.out_channels, self._kernels, norm=self._norm,
                    activation=self._activation, residual=use_residual)
                flat_size += self._down3.out_channels * 4
                self._ss3 = SpatialSoftmax3D(
                    spatial_size, spatial_size, spatial_size,
                    self._down3.out_channels)
                self._up3 = upsample_block_class(
                    self._kernels, self._kernels, scale_factor=2, norm=self._norm,
                    activation=self._activation, residual=use_residual)
            self._up2 = upsample_block_class(
                k2, self._kernels, scale_factor=2, norm=self._norm,
                activation=self._activation, residual=use_residual)

        if self._use_context and self._dev_cfgs.get('cat_up1', False):
            k1 = k1 + self._encode_context_size if self._encode_context else k1 + self._inp_context_size
            
        self._up1 = upsample_block_class(
            k1, self._kernels, scale_factor=2, norm=self._norm,
            activation=self._activation, residual=use_residual)

        self._global_maxp = nn.AdaptiveMaxPool3d(1)
        self._local_maxp = nn.MaxPool3d(3, 2, padding=1)

        final_ins = self._kernels * 2
        if self._use_context and self._dev_cfgs.get('cat_f1', False):
            final_ins = final_ins + self._encode_context_size if self._encode_context else final_ins + self._inp_context_size
         
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

        if self._use_context and self._dev_cfgs.get('cat_final', False):
            flat_size = flat_size + self._encode_context_size if self._encode_context else flat_size + self._inp_context_size


        if self._out_dense > 0:
            self._dense0 = DenseBlock(
                flat_size, self._dense_feats, None, self._activation)
            self._dense1 = DenseBlock(
                self._dense_feats, self._dense_feats, None, self._activation)
            self._dense2 = DenseBlock(
                self._dense_feats, self._out_dense, None, None)

        if self._use_context and self._dev_cfgs.get('classify', False): # NOTE: use self._encode_context_size as hidden dim! 
            self._classify_mlp = DenseBlock(
                    self._inp_context_size, 
                    self._encode_context_size, None, self._activation)
            self._classify_mlp_2 = DenseBlock(
                    self._encode_context_size, 10, None, self._activation)
 
    def forward(self, ins, proprio, prev_layer_voxel_grid, context):
        b, _, d, h, w = ins.shape # b, 10, 16, 16, 16
        
        if len(context.shape) == 1: # for acting
            context = rearrange(context, 'c -> 1 c')
        if self._dev_cfgs.get('discretise', False):
            context = sample_gumbel_softmax(context)

        x = self._input_preprocess(ins)

        if self._include_prev_layer:
            y = self._input_preprocess_prev_layer(prev_layer_voxel_grid)
            x = torch.cat([x, y], dim=1)

        if self._low_dim_size > 0:
            p = self._proprio_preprocess(proprio)
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, d, h, w)
            x = torch.cat([x, p], dim=1)
        
        if self._use_context and self._encode_context:
            if self._dev_cfgs.get('ctxt_conv3d', False):
                context = self.ctxt_activate(
                    self.ctxt_conv3d(context)
                    )
                context = rearrange(context, 'bk d n h w -> bk (d n h w)')
            ctxt = self._context_preprocess(context) # b, 64 

            if self._encode_context_hidden > 0:
                ctxt = self._context_preprocess_2(ctxt)
        else:
            ctxt = context 
        # print('networks Qnet forward: ctxt shape', ctxt.shape)
        rep0 = repeat(ctxt, 'b c -> b c d h w', d=d, h=h, w=w)
        # print('networks Qnet forward: repeated shape', rep0.shape) # b, 64, 16, 16, 16
        down0_in = torch.cat([x, rep0], dim=1) if self._use_context else x 

        d0 = self._down0(down0_in)
        # print('forward Qnet: d0 shape', d0.shape) #[b, 64, 16, 16, 16]
        ss0 = self._ss0(d0)
        maxp0 = self._global_maxp(d0).view(b, -1)
        down1_in = self._local_maxp(d0)
        # print('forward Qnet: maxp0 shape', maxp0.shape) #[b, 64]
        # print('forward Qnet: down1_in shape', down1_in.shape) #[b, 64, 8, 8, 8]
        
        if self._use_context and self._dev_cfgs.get('cat_down1', False):
            _, _, dd, hh, ww = down1_in.shape
            rep1 = repeat(ctxt, 'b c -> b c d h w', d=dd, h=hh, w=ww)
            down1_in = torch.cat([down1_in, rep1], dim=1)
            #print('forward Qnet : cated down1_in shape', down1_in.shape)
        d1 = u = self._down1(down1_in)
        ss1 = self._ss1(d1)
        maxp1 = self._global_maxp(d1).view(b, -1)
        # print('forward Qnet: maxp1 shape', maxp1.shape) #[b, 128]

        feats = [ss0, maxp0, ss1, maxp1]

        if self._voxel_size > 8:
            down2_in = self._local_maxp(d1)
            # print('forward Qnet: down2_in shape', down2_in.shape) # [b, 128, 4, 4, 4]
            if self._use_context and self._dev_cfgs.get('cat_down2', False):
                _, _, dd, hh, ww = down2_in.shape
                rep2 = repeat(ctxt, 'b c -> b c d h w', d=dd, h=hh, w=ww)
                down2_in = torch.cat([down2_in, rep2], dim=1)
                # print('forward Qnet : cated down2_in shape', down2_in.shape)
            d2 = u = self._down2(down2_in)
            # print('forward Qnet: down2_out shape', d2.shape) # [b, 128, 4, 4, 4]
            feats.extend([self._ss2(d2), self._global_maxp(d2).view(b, -1)])
            if self._voxel_size > 16:
                d3 = self._down3(self._local_maxp(d2))
                feats.extend([self._ss3(d3), self._global_maxp(d3).view(b, -1)])
                u3 = self._up3(d3)
                u = torch.cat([d2, u3], dim=1) 
            
            up2_in = u
            # print('forward Qnet: up2_in shape', up2_in.shape) # [b, 256, 4, 4, 4]
            if self._use_context and self._dev_cfgs.get('cat_up2', False):
                _, _, dd, hh, ww = up2_in.shape
                rep22 = repeat(ctxt, 'b c -> b c d h w', d=dd, h=hh, w=ww)
                up2_in = torch.cat([up2_in, rep22], dim=1)
                # print('forward Qnet : cated up2_in shape', up2_in.shape)
            u2 = self._up2(up2_in)
            # print('forward Qnet: up2_out shape', u2.shape) # torch.Size([b, 64, 8, 8, 8])
            u = torch.cat([d1, u2], dim=1)

        # print('forward Qnet: up1_in shape', u.shape) [b, 192, 8, 8, 8]
        up1_in = u
        if self._use_context and self._dev_cfgs.get('cat_up1', False):
            _, _, dd, hh, ww = up1_in.shape
            rep11 = repeat(ctxt, 'b c -> b c d h w', d=dd, h=hh, w=ww)
            up1_in = torch.cat([up1_in, rep11], dim=1)
            # print('forward Qnet : cated up1_in shape', up1_in.shape)
        u1 = self._up1(up1_in)

        f1_in = torch.cat([d0, u1], dim=1)
        # print('forward Qnet: f1 shape', f1_in.shape) #[b, 64, 16, 16, 16]
        if self._use_context and self._dev_cfgs.get('cat_f1', False):
            _, _, dd, hh, ww = f1_in.shape
            repf1 = repeat(ctxt, 'b c -> b c d h w', d=dd, h=hh, w=ww)
            f1_in = torch.cat([f1_in, repf1], dim=1)
            # print('forward Qnet : cated f1_in shape', f1_in.shape)
        f1 = self._final(f1_in)
        
        trans = self._final2(f1)
        # print('forward Qnet: trans shape', trans.shape) #[b, 1, 16, 16, 16]

        feats.extend([self._ss_final(f1), self._global_maxp(f1).view(b, -1)])

        self.latent_dict = {
            'd0': d0.mean(-1).mean(-1).mean(-1),
            'd1': d1.mean(-1).mean(-1).mean(-1),
            'u1': u1.mean(-1).mean(-1).mean(-1),
            'trans_out': trans,
            'encoded_context': ctxt,
        }

        rot_and_grip_out = None
        if self._out_dense > 0:
            dense0_in = torch.cat(feats, 1)
            # print('forward Qnet: dense0_in shape', dense0_in.shape) [b, 2048]
            dense0 = self._dense0(dense0_in)
            dense1 = self._dense1(dense0)
            rot_and_grip_out = self._dense2(dense1)
            self.latent_dict.update({
                'dense0': dense0,
                'dense1': dense1,
                'dense2': rot_and_grip_out,
            })

        if self._voxel_size > 8:
            self.latent_dict.update({
                'd2': d2.mean(-1).mean(-1).mean(-1),
                'u2': u2.mean(-1).mean(-1).mean(-1),
            })
        if self._voxel_size > 16:
            self.latent_dict.update({
                'd3': d3.mean(-1).mean(-1).mean(-1),
                'u3': u3.mean(-1).mean(-1).mean(-1),
            })
        
        return trans, rot_and_grip_out, ctxt 

    def classify_only(self, context):
        pred = self._classify_mlp(context)
        pred = self._classify_mlp_2(pred)
        return pred 

class Qattention3DNetWithFiLM(Qattention3DNet):
    """For each Conv3DInceptionBlock, film-encode the context to 
    outuput (beta, gamma) for weight+shift the features """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_dense: int,
                 voxel_size: int,
                 low_dim_size: int,
                 kernels: int,
                 inp_context_size: int, 
                 use_context: bool = True,
                 norm: str = None,
                 activation: str = 'relu',
                 dense_feats: int = 32,
                 include_prev_layer = False, # has been false  
                 dev_cfgs: dict = {},   
                 ):
        super(Qattention3DNet, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._norm = norm
        self._activation = activation
        self._kernels = kernels
        self._low_dim_size = low_dim_size
        self._build_calls = 0
        self._voxel_size = voxel_size
        self._dense_feats = dense_feats
        self._out_dense = out_dense
        self._include_prev_layer = include_prev_layer
        # context related
        # self._use_prev_context = use_prev_context 
        self._inp_context_size = inp_context_size 
        self._use_context = use_context
        self._dev_cfgs = dev_cfgs
        print('Using in-development cfgs:', dev_cfgs) 
        print(f'Qattention3DNet: Input context embedding size: {inp_context_size}')

    
    def build(self):
        block_class = Conv3DInceptionBlock
        upsample_block_class = Conv3DInceptionBlockUpsampleBlock
        if self._dev_cfgs.get('conv3d', False):
            block_class = partial(Conv3DBlock, kernel_sizes=3, strides=1)
             # Conv3DResNetBlock 
            upsample_block_class = partial(Conv3DUpsampleBlock,
                kernel_sizes=3, strides=1)
        
        use_residual = False
        self._build_calls += 1
        if self._build_calls != 1:
            raise RuntimeError('Build needs to be called once.')

        spatial_size = self._voxel_size
        self._input_preprocess = block_class(
            self._in_channels, self._kernels, norm=self._norm,
            activation=self._activation)

        d0_ins = self._input_preprocess.out_channels
        # create film layer for every set of feature maps output
        self._d0_film = DenseBlock(self._inp_context_size, d0_ins * 2, None, self._activation)
        # Note film happens *before* low_dim get concated
        if self._include_prev_layer:
            PREV_VOXEL_CHANNELS = 0
            self._input_preprocess_prev_layer = block_class(
                self._in_channels + PREV_VOXEL_CHANNELS, self._kernels, norm=self._norm,
                activation=self._activation)
            d0_ins += self._input_preprocess_prev_layer.out_channels

        if self._low_dim_size > 0:
            self._proprio_preprocess = DenseBlock(
                self._low_dim_size, self._kernels, None, self._activation)
            d0_ins += self._kernels

        # Note context size can also equal _kernels if this is intermediate net and use_prev_context is True 
         
        self._down0 = block_class(
            d0_ins, self._kernels, norm=self._norm,
            activation=self._activation, residual=use_residual)
        self._ss0 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down0.out_channels)
        spatial_size //= 2

        d1_ins = self._down0.out_channels
        self._d1_film = DenseBlock(self._inp_context_size, d1_ins * 2, None, self._activation)
        self._down1 = block_class(
            d1_ins, self._kernels * 2, norm=self._norm,
            activation=self._activation, residual=use_residual)
        self._ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down1.out_channels)
        spatial_size //= 2

        flat_size = self._down0.out_channels * 4 + self._down1.out_channels * 4
        
        k1 = self._down1.out_channels
        if self._voxel_size > 8:
            k1 += self._kernels
            d2_ins = self._down1.out_channels
            self._d2_film = DenseBlock(self._inp_context_size, d2_ins * 2, None, self._activation)
            self._down2 = block_class(
                d2_ins, self._kernels * 4, norm=self._norm,
                activation=self._activation,  residual=use_residual)
            flat_size += self._down2.out_channels * 4
            self._ss2 = SpatialSoftmax3D(
                spatial_size, spatial_size, spatial_size,
                self._down2.out_channels)
            spatial_size //= 2
            
            k2 = self._down2.out_channels
            if self._voxel_size > 16:
                raise ValueError # not worrying for now 
                k2 *= 2
                self._down3 = block_class(
                    self._down2.out_channels, self._kernels, norm=self._norm,
                    activation=self._activation, residual=use_residual)
                flat_size += self._down3.out_channels * 4
                self._ss3 = SpatialSoftmax3D(
                    spatial_size, spatial_size, spatial_size,
                    self._down3.out_channels)
                self._up3 = upsample_block_class(
                    self._kernels, self._kernels, scale_factor=2, norm=self._norm,
                    activation=self._activation, residual=use_residual)
            
            self._u2_film = DenseBlock(self._inp_context_size, k2 * 2, None, self._activation)
            self._up2 = upsample_block_class(
                k2, self._kernels, scale_factor=2, norm=self._norm,
                activation=self._activation, residual=use_residual)

        self._u1_film = DenseBlock(self._inp_context_size, k1 * 2, None, self._activation)
        self._up1 = upsample_block_class(
            k1, self._kernels, scale_factor=2, norm=self._norm,
            activation=self._activation, residual=use_residual)

        self._global_maxp = nn.AdaptiveMaxPool3d(1)
        self._local_maxp = nn.MaxPool3d(3, 2, padding=1)

        final_ins = self._kernels * 2
        self._f1_film = DenseBlock(self._inp_context_size, final_ins * 2, None, self._activation)
        self._final = Conv3DBlock(
            final_ins, self._kernels, kernel_sizes=3,
            strides=1, norm=self._norm, activation=self._activation)
        self._f2_film = DenseBlock(self._inp_context_size, self._kernels * 2, None, self._activation)
        self._final2 = Conv3DBlock(
            self._kernels, self._out_channels, kernel_sizes=3,
            strides=1, norm=None, activation=None)

        self._ss_final = SpatialSoftmax3D(
            self._voxel_size, self._voxel_size, self._voxel_size,
            self._kernels)
        flat_size += self._kernels * 4 

        if self._out_dense > 0:
            self._dense0 = DenseBlock(
                flat_size, self._dense_feats, None, self._activation)
            self._dense1 = DenseBlock(
                self._dense_feats, self._dense_feats, None, self._activation)
            self._dense2 = DenseBlock(
                self._dense_feats, self._out_dense, None, None)
 
    def forward(self, ins, proprio, prev_layer_voxel_grid, context):
        b, _, d, h, w = ins.shape
        if len(context.shape) == 1: # for acting
            context = rearrange(context, 'c -> 1 c')

        x = self._input_preprocess(ins)
        if self._use_context:
            
            gam, beta = self._d0_film(context).split(x.shape[1], dim=1)
            gam, beta = repeat(gam, 'b c -> b c 1 1 1'), repeat(beta, 'b c -> b c 1 1 1')
            assert x.shape == (gam * x + beta).shape
            x = gam * x + beta
        if self._include_prev_layer:
            y = self._input_preprocess_prev_layer(prev_layer_voxel_grid)
            x = torch.cat([x, y], dim=1)

        if self._low_dim_size > 0:
            p = self._proprio_preprocess(proprio)
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, d, h, w)
            x = torch.cat([x, p], dim=1) 
        
        
        d0 = self._down0(x)
        # print('forward Qnet: d0 shape', d0.shape) #[b, 64, 16, 16, 16]
        # do film before the softmax/max pools
        if self._use_context:
            gam, beta = self._d1_film(context).split(d0.shape[1], dim=1)
            gam, beta = repeat(gam, 'b c -> b c 1 1 1'), repeat(beta, 'b c -> b c 1 1 1')
            assert d0.shape == (gam * d0 + beta).shape, f"Got before/after shape: {d0.shape}, {(gam * d0 + beta).shape}"
            d0 = d0 * gam + beta 

        ss0 = self._ss0(d0)
        maxp0 = self._global_maxp(d0).view(b, -1)
        down1_in = self._local_maxp(d0)
        # print('forward Qnet: maxp0 shape', maxp0.shape) #[b, 64]
        # print('forward Qnet: down1_in shape', down1_in.shape) #[b, 64, 8, 8, 8]
        
        d1 = u = self._down1(down1_in)
        if self._use_context:
            gam, beta = self._d2_film(context).split(d1.shape[1], dim=1)
            gam, beta = repeat(gam, 'b c -> b c 1 1 1'), repeat(beta, 'b c -> b c 1 1 1')
            assert d1.shape == (gam * d1 + beta).shape, f"Got before/after shape: {d1.shape}, {(gam * d1 + beta).shape}"
            d1 = d1 * gam + beta

        ss1 = self._ss1(d1)
        maxp1 = self._global_maxp(d1).view(b, -1)
        # print('forward Qnet: maxp1 shape', maxp1.shape) #[b, 128]

        feats = [ss0, maxp0, ss1, maxp1]

        if self._voxel_size > 8:
            down2_in = self._local_maxp(d1)
            # print('forward Qnet: down2_in shape', down2_in.shape) # [b, 128, 4, 4, 4]
            d2 = u = self._down2(down2_in) 
            # print('forward Qnet: down2_out shape', d2.shape) # [b, 128, 4, 4, 4]
            feats.extend([self._ss2(d2), self._global_maxp(d2).view(b, -1)])
            if self._voxel_size > 16:
                raise NotImplementedError 
                d3 = self._down3(self._local_maxp(d2))
                feats.extend([self._ss3(d3), self._global_maxp(d3).view(b, -1)])
                u3 = self._up3(d3)
                u = torch.cat([d2, u3], dim=1) 
            
            up2_in = u
            if self._use_context:
                gam, beta = self._u2_film(context).split(up2_in.shape[1], dim=1)
                gam, beta = repeat(gam, 'b c -> b c 1 1 1'), repeat(beta, 'b c -> b c 1 1 1')
                assert up2_in.shape == (gam * up2_in + beta).shape 
                up2_in = up2_in * gam + beta
            u2 = self._up2(up2_in)
            # print('forward Qnet: up2_out shape', u2.shape) # torch.Size([b, 64, 8, 8, 8])
            u = torch.cat([d1, u2], dim=1)

        # print('forward Qnet: up1_in shape', u.shape) [b, 192, 8, 8, 8]
        up1_in = u
        if self._use_context:
            gam, beta = self._u1_film(context).split(up1_in.shape[1], dim=1)
            gam, beta = repeat(gam, 'b c -> b c 1 1 1'), repeat(beta, 'b c -> b c 1 1 1')
            assert up1_in.shape == (gam * up1_in + beta).shape 
            up1_in = up1_in * gam + beta 
        u1 = self._up1(up1_in)

        f1_in = torch.cat([d0, u1], dim=1)
        # print('forward Qnet: f1 shape', f1_in.shape) #[b, 64, 16, 16, 16]
        if self._use_context:
            gam, beta = self._f1_film(context).split(f1_in.shape[1], dim=1)
            gam, beta = repeat(gam, 'b c -> b c 1 1 1'), repeat(beta, 'b c -> b c 1 1 1')
            assert f1_in.shape == (gam * f1_in + beta).shape 
            f1_in = f1_in * gam + beta  
        f1 = self._final(f1_in)
        
        if self._use_context:
            gam, beta = self._f2_film(context).split(f1.shape[1], dim=1)
            gam, beta = repeat(gam, 'b c -> b c 1 1 1'), repeat(beta, 'b c -> b c 1 1 1')
            f1 = f1 * gam + beta 
         
        trans = self._final2(f1)
        # print('forward Qnet: trans shape', trans.shape) #[b, 1, 16, 16, 16]

        feats.extend([self._ss_final(f1), self._global_maxp(f1).view(b, -1)])

        self.latent_dict = {
            'd0': d0.mean(-1).mean(-1).mean(-1),
            'd1': d1.mean(-1).mean(-1).mean(-1),
            'u1': u1.mean(-1).mean(-1).mean(-1),
            'trans_out': trans, 
        }

        rot_and_grip_out = None
        if self._out_dense > 0:
            dense0_in = torch.cat(feats, 1)
            # print('forward Qnet: dense0_in shape', dense0_in.shape) [b, 2048]
            dense0 = self._dense0(dense0_in)
            dense1 = self._dense1(dense0)
            rot_and_grip_out = self._dense2(dense1)
            self.latent_dict.update({
                'dense0': dense0,
                'dense1': dense1,
                'dense2': rot_and_grip_out,
            })

        if self._voxel_size > 8:
            self.latent_dict.update({
                'd2': d2.mean(-1).mean(-1).mean(-1),
                'u2': u2.mean(-1).mean(-1).mean(-1),
            })
        if self._voxel_size > 16:
            self.latent_dict.update({
                'd3': d3.mean(-1).mean(-1).mean(-1),
                'u3': u3.mean(-1).mean(-1).mean(-1),
            })
        
        return trans, rot_and_grip_out, None  # in place of encoded_context

class PEARLContextEncoder(nn.Module):
    """Encode 1 or 2 observation frames, concat with 8-dim action and 1-dim reward"""
    def __init__(self,  
                 encode_next_obs: bool = True, 
                 conv_kernel_sizes: bool = [3],
                 conv_out_channels: List[int] = [32],
                 strides: List[int] = [3],
                 action_size: int = 8,
                 norm: str = None,
                 activation: str = 'lrelu',
                 mlp_hidden_sizes: list = [128], 
                 output_size: int = 32
                 ):
        super(PEARLContextEncoder, self).__init__()
        self._encode_next_obs = encode_next_obs
        conv_block = Conv3DBlock if encode_next_obs else Conv2DBlock
        assert len(conv_kernel_sizes) == len(conv_out_channels) == len(strides), 'Must specify same length'
        conv_blocks = []
        for i, kernel in enumerate(conv_kernel_sizes):
            block = conv_block(
                in_channels=(3 if i == 0 else conv_out_channels[i-1]), 
                out_channels=conv_out_channels[i], 
                kernel_sizes=kernel, 
                activation=activation,
                strides=strides[i],
                )
            conv_blocks.append(block)  
        self._conv_blocks = nn.Sequential(*conv_blocks)
        dummy_inp = torch.zeros(1, 3, 2, IMG_SIZE, IMG_SIZE)  if encode_next_obs else torch.zeros(1, 3, IMG_SIZE, IMG_SIZE) 
         
        dummy_out = self._conv_blocks(dummy_inp)
        
        conv_out_size = torch.flatten(dummy_out, start_dim=1).shape[-1]
        print(f'Building Context Encoder MLP with conv flattened input size {conv_out_size}, output size {output_size}')
        
        conv_out_size += action_size +  1
        mlps = []
        for i, hidden_size in enumerate(mlp_hidden_sizes):
            mlps.append(
                DenseBlock(
                    in_features=conv_out_size if i == 0 else mlp_hidden_sizes[i-1],
                    out_features=hidden_size,
                    norm=None,
                    activation=activation,
                )
            )
        mlps.append(
            DenseBlock(
                in_features=mlp_hidden_sizes[-1],
                out_features=output_size * 2, # need to predict both mean and std 
                norm=None,
                activation=activation,
            )
        )
        self._mlps = nn.Sequential(*mlps)
        self.latent_size = output_size 
        self._device = None

    def forward(self, replay_sample):
        conv_in = replay_sample['context_obs'].to(self._device)  
        assert conv_in.shape[1] == 3, 'Must have shape b, channel, n_frames, h, w'
        if not self._encode_next_obs:
            conv_in = conv_in[:, :, 0]
        conv_out = self._conv_blocks(conv_in) 
        conv_out = torch.cat(
                [
                    conv_out.flatten(start_dim=1), 
                    replay_sample['context_action'].to(self._device) ,
                    replay_sample['context_reward'].to(self._device) 
                ], dim=1)
        
        mlp_out = self._mlps(conv_out) 
         
        return mlp_out

    def set_device(self, device):
        self._device = device 
        self.to(device)

if __name__ == '__main__':
    # qnet = Qattention3DNetWithContext(
    #     in_channels=10,
    #     out_channels=1,
    #     voxel_size=16,
    #     out_dense=0,
    #     kernels=64,
    #     norm=None, 
    #     dense_feats=128,
    #     activation='lrelu',
    #     low_dim_size=10,
    #     inp_context_size=20,
    #     dev_cfgs={'conv3d':False},
    #     )
    # qnet.build()
    # b = 1
    # ins = torch.ones(b, 10 , 3,128,128)
    # prop = torch.ones(b,10)
    # context = torch.ones(b, 20)
    # print(qnet(ins, prop, None, context).shape)
    # num_params = sum([p.numel() for p in qnet.parameters() if p.requires_grad])
    # print('Qattention3DNetWithContext # of params:', num_params)

    # qnet = Qattention3DNetWithContext(
    #     in_channels=10,
    #     out_channels=1,
    #     voxel_size=16,
    #     out_dense=0,
    #     kernels=64,
    #     norm=None, 
    #     dense_feats=128,
    #     activation='lrelu',
    #     low_dim_size=10,
    #     inp_context_size=20,
    #     dev_cfgs={'conv3d': True},
    #     )
    # qnet.build()
    # num_params = sum([p.numel() for p in qnet.parameters() if p.requires_grad])
    # print('Qattention3DNetWithContext # of params:', num_params)

    # qnet = Qattention3DNetWithFiLM(
    #     in_channels=10,
    #     out_channels=1,
    #     voxel_size=16,
    #     out_dense=0,
    #     kernels=64,
    #     norm=None, 
    #     dense_feats=128,
    #     activation='lrelu',
    #     low_dim_size=10,
    #     use_context=False,
    #     inp_context_size=20,
    #     dev_cfgs={'conv3d': False},
    #     )
    # qnet.build()
    # b = 1
    # ins = torch.ones(b, 10, 3,128,128)
    # prop = torch.ones(b,10)
    # context = torch.ones(b, 20)
    # print(qnet(ins, prop, None, context).shape)
    # num_params = sum([p.numel() for p in qnet.parameters() if p.requires_grad])
    # print('Qattention3DNetWithFiLM # of params:', num_params)
    encoder = PEARLContextEncoder(
        encode_next_obs=False, 
        conv_kernel_sizes=[3, 3],
        conv_out_channels=[32, 32],
        strides=[2,2],
        action_size=8,
        norm=None,
        activation='lrelu',
        mlp_hidden_sizes=[256, 128], 
        output_size=16
        )
    inp = {
        'context_obs': torch.ones(6, 3, 2, IMG_SIZE, IMG_SIZE),
        'context_action': torch.ones(6, 8),
        'context_reward': torch.ones(6, 1)
    } 
    out = encoder(inp) # B*K, 32
    batch = 3
    out = rearrange(out, '(b k) d -> b k d', b=batch)
    mus, sigmas = out[:,:,:16], F.softplus(out[:,:,16:])


    def _product_of_gaussians(mus, sigmas_squared):
        '''
        compute mu, sigma of product of gaussians
        '''
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
        return mu, sigma_squared
   
    mu, sigma_squared = _product_of_gaussians(mus[0], sigmas[0])
    print(mu.shape, sigma_squared.shape)
    print([m.shape for m in torch.unbind(mus)])
    z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mus), torch.unbind(sigmas))]
    z_means = torch.stack([p[0] for p in z_params])
    z_vars = torch.stack([p[1] for p in z_params])

    posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) \
            for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
    z = [d.rsample() for d in posteriors]
    z = torch.stack(z)
    print(z.shape)
    def compute_kl_loss(z_means, z_vars):
        """ compute KL( q(z|c) || r(z) ) """
        prior = torch.distributions.Normal(
            torch.zeros(16), torch.ones(16))
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) \
            for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum
    print(compute_kl_loss(z_means, z_vars))