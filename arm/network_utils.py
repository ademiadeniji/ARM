import copy
from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.02 

def act_layer(act):
    if act == 'relu':
        return nn.ReLU()
    if act == 'inp_relu':
        return nn.ReLU(inplace=True)
    elif act == 'lrelu':
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == 'elu':
        return nn.ELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError('%s not recognized.' % act)

def norm_layer3d(norm, channels):
    if norm == 'batch':
        return nn.BatchNorm3d(channels)
    else:
        raise ValueError('%s not recognized.' % norm)

def norm_layer2d(norm, channels):
    if norm == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == 'layer':
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == 'group':
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError('%s not recognized.' % norm)


def norm_layer1d(norm, num_channels):
    if norm == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm == 'instance':
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == 'layer':
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError('%s not recognized.' % norm)


class Conv2DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None, padding_mode='replicate'):
        super(Conv2DBlock, self).__init__()
        padding = kernel_sizes // 2 if isinstance(kernel_sizes, int) else (
            kernel_sizes[0] // 2, kernel_sizes[1] // 2)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv2d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv2d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv2d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None):
        super(Conv2DUpsampleBlock, self).__init__()
        layer = [Conv2DBlock(
            in_channels, out_channels, kernel_sizes, 1, norm, activation)]
        if strides > 1:
            layer.append(nn.Upsample(
                scale_factor=strides, mode='bilinear',
                align_corners=False))
        convt_block = Conv2DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation)
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class Conv3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_sizes: Union[int, list], strides,
                 norm=None, activation=None, padding_mode='replicate',
                 padding=None, residual=False):
        super(Conv3DBlock, self).__init__() 
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'relu' or activation == 'inp_relu':
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer3d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels =  out_channels

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv3DUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_sizes=3, strides=1,
                 norm=None, activation=None, residual=False):
        super(Conv3DUpsampleBlock, self).__init__()
        layer = [Conv3DBlock(
            in_channels, out_channels, kernel_sizes, strides, norm, activation)]
        if scale_factor > 1:
            layer.append(nn.Upsample(
                scale_factor=scale_factor, mode='trilinear',
                align_corners=False))
        convt_block = Conv3DBlock(
            out_channels, out_channels, kernel_sizes, strides, norm, activation)
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class DenseBlock(nn.Module):

    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.linear.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.linear.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.linear.weight, a=LRELU_SLOPE, nonlinearity='leaky_relu')
            nn.init.zeros_(self.linear.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class SiameseNet(nn.Module):

    def __init__(self,
                 input_channels: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 activation: str = 'relu'):
        super(SiameseNet, self).__init__()
        self._input_channels = input_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self.output_channels = filters[-1] #* len(input_channels)

    def build(self):
        self._siamese_blocks = nn.ModuleList()
        for i, ch in enumerate(self._input_channels):
            blocks = []
            for i, (filt, ksize, stride) in enumerate(
                    zip(self._filters, self._kernel_sizes, self._strides)):
                conv_block = Conv2DBlock(
                    ch, filt, ksize, stride, self._norm, self._activation)
                blocks.append(conv_block)
            self._siamese_blocks.append(nn.Sequential(*blocks))
        self._fuse = Conv2DBlock(self._filters[-1] * len(self._siamese_blocks),
                                 self._filters[-1], 1, 1, self._norm,
                                 self._activation)

    def forward(self, x):
        if len(x) != len(self._siamese_blocks):
            raise ValueError('Expected a list of tensors of size %d.' % len(
                self._siamese_blocks))
        self.streams = [stream(y) for y, stream in zip(x, self._siamese_blocks)]
        y = self._fuse(torch.cat(self.streams, 1))
        return y


class CNNAndFcsNet(nn.Module):

    def __init__(self,
                 siamese_net: SiameseNet,
                 low_dim_state_len: int,
                 input_resolution: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 fc_layers: List[int] = None,
                 activation: str = 'relu'):
        super(CNNAndFcsNet, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._input_resolution = input_resolution

    def build(self):
        self._siamese_net.build()
        layers = []
        channels = self._input_channels
        for i, (filt, ksize, stride) in enumerate(
                list(zip(self._filters, self._kernel_sizes, self._strides))[
                :-1]):
            layers.append(Conv2DBlock(
                channels, filt, ksize, stride, self._norm, self._activation))
            channels = filt
        layers.append(Conv2DBlock(
            channels, self._filters[-1], self._kernel_sizes[-1],
            self._strides[-1]))
        self._cnn = nn.Sequential(*layers)
        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(
            DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)

    def forward(self, observations, low_dim_ins):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)
        x = self._cnn(combined)
        x = self._maxp(x).squeeze(-1).squeeze(-1)
        return self._fcs(x)


class Conv3DInceptionBlockUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 norm=None, activation=None, residual=False):
        super(Conv3DInceptionBlockUpsampleBlock, self).__init__()
        layer = []

        convt_block = Conv3DInceptionBlock(
            in_channels, out_channels, norm, activation)
        layer.append(convt_block)

        if scale_factor > 1:
            layer.append(nn.Upsample(
                scale_factor=scale_factor, mode='trilinear',
                align_corners=False))

        convt_block = Conv3DInceptionBlock(
            out_channels, out_channels, norm, activation)
        layer.append(convt_block)

        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class Conv3DInceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm=None, activation=None,
                 residual=False):
        super(Conv3DInceptionBlock, self).__init__()
        self._residual = residual
        cs = out_channels // 4
        assert out_channels % 4 == 0
        latent = 32
        self._1x1conv = Conv3DBlock(
            in_channels, cs * 2, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)

        self._1x1conv_a = Conv3DBlock(
            in_channels, latent, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)
        self._3x3conv = Conv3DBlock(
            latent, cs, kernel_sizes=3, strides=1,
            norm=norm, activation=activation)

        self._1x1conv_b = Conv3DBlock(
            in_channels, latent, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)
        self._5x5_via_3x3conv_a = Conv3DBlock(
            latent, latent, kernel_sizes=3, strides=1, norm=norm,
            activation=activation)
        self._5x5_via_3x3conv_b = Conv3DBlock(
            latent, cs, kernel_sizes=3, strides=1, norm=norm,
            activation=activation)
        self.out_channels = out_channels + (in_channels if residual else 0)

    def forward(self, x):
        yy = []
        if self._residual:
            yy = [x]
        return torch.cat(yy + [self._1x1conv(x),
                               self._3x3conv(self._1x1conv_a(x)),
                               self._5x5_via_3x3conv_b(self._5x5_via_3x3conv_a(
                                   self._1x1conv_b(x)))], 1)


class Conv3DResNetBlock(nn.Module):
    """ 
    ResNet blocks are simpler: no concat, just 3 layers of convolution and residual connection 
    3dBatchnorm and Inplace Relu are defaults from SlowFast 
    """
    def __init__(self, in_channels, out_channels, norm="batch", activation="inp_relu",
                 style='bottleneck', residual=True):
        super(Conv3DResNetBlock, self).__init__()
        if style == 'basic':
            # smaller block, just two 3x3convs
            latent = out_channels // 2
            layers = [
                Conv3DBlock(in_channels, latent, kernel_sizes=3, strides=1, 
                    norm=norm, activation=activation),
                Conv3DBlock(latent, out_channels, kernel_sizes=3, strides=1, 
                    norm=norm, activation=activation),
            ] 
        elif style == 'bottleneck':
            latent = out_channels // 1
            layers = [
                Conv3DBlock(in_channels, out_channels // 2, kernel_sizes=1, strides=1, 
                    norm=norm, activation=activation),
                Conv3DBlock(out_channels // 2, out_channels, kernel_sizes=3, strides=1, 
                    norm=norm, activation=activation),
                Conv3DBlock(out_channels, out_channels, kernel_sizes=1, strides=1, 
                    norm=norm, activation=activation)
                    ]
        else:
            raise ValueError('%s not recognized.' % style)
        
        self.activate = act_layer(activation)
        self._layers = nn.Sequential(*layers)
        self.out_channels = out_channels   

    def forward(self, x):
        out = self._layers(x)
        return self.activate(x + out)


class Conv3DResNetUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor,
                  norm="batch", activation="inp_relu",  style='bottleneck', residual=True):
        super(Conv3DResNetUpsampleBlock, self).__init__()
        layer = [] 
        convt_block = Conv3DResNetBlock(
            in_channels, out_channels, norm, activation, style)
        layer.append(convt_block)

        if scale_factor > 1:
            layer.append(nn.Upsample(
                scale_factor=scale_factor, mode='trilinear',
                align_corners=False))

        convt_block = Conv3DResNetBlock(
            out_channels, out_channels, norm, activation, style)
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class SpatialSoftmax3D(torch.nn.Module):

    def __init__(self, depth, height, width, channel):
        super(SpatialSoftmax3D, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = 0.01
        pos_x, pos_y, pos_z = np.meshgrid(
            np.linspace(-1., 1., self.depth),
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(
            pos_x.reshape(self.depth * self.height * self.width)).float()
        pos_y = torch.from_numpy(
            pos_y.reshape(self.depth * self.height * self.width)).float()
        pos_z = torch.from_numpy(
            pos_z.reshape(self.depth * self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)

    def forward(self, feature):
        feature = feature.view(
            -1, self.height * self.width * self.depth)  # (B, c*d*h*w)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1,
                               keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1,
                               keepdim=True)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1,
                               keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 3)
        return feature_keypoints


class SiameseCNNModel(nn.Module):
    """Note(Mandi): migrated from QAttentionMultitask, use this for context embedder"""
    def __init__(self,
                 input_shapes: List[List[int]],
                 pre_filters: List[int],
                 pre_kernel_sizes: List[int],
                 pre_strides: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 activation: str = 'relu',
                 activation_on_last: str = None):
        super(SiameseCNNModel, self).__init__()
        self._input_shapes = input_shapes
        self._pre_filters = pre_filters
        self._pre_kernel_sizes = pre_kernel_sizes
        self._pre_strides = pre_strides
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._activation_on_last = activation_on_last
        self._build_calls = 0

    def build(self):
        self._build_calls += 1
        if self._build_calls != 1:
            raise RuntimeError('Build needs to be called once.')
        self._siamese_block = SiameseBlock(
            self._input_shapes, self._pre_filters, self._pre_kernel_sizes,
            self._pre_strides, self._norm, self._activation)
        self._cnn = CNNTrunkBlock(
            self._siamese_block.out_shape, self._filters, self._kernel_sizes,
            self._strides, self._norm, self._activation,
            self._activation_on_last)

    def forward(self, x):
        y = self._cnn(self._siamese_block(x))
        self.streams = self._siamese_block.streams
        return y


class SiameseCNNWithFCModel(SiameseCNNModel):

    def __init__(self,
                 input_shapes: List[List[int]],
                 pre_filters: List[int],
                 pre_kernel_sizes: List[int],
                 pre_strides: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 fc_layers: List[int],
                 norm: str = None,
                 activation: str = 'relu'):
        super(SiameseCNNWithFCModel, self).__init__(
            input_shapes, pre_filters, pre_kernel_sizes, pre_strides,
            filters, kernel_sizes, strides, norm, activation, activation
        )
        self._maxp = nn.AdaptiveMaxPool2d(1)
        self._mlp = MLPModel(
            filters[-1], fc_layers, activation, None, None)

    def build(self):
        super(SiameseCNNWithFCModel, self).build()
        self._mlp.build()

    def forward(self, x):
        y = super(SiameseCNNWithFCModel, self).forward(x)
        y = self._maxp(y).squeeze(-1).squeeze(-1)
        y = self._mlp(y)
        return y

if __name__ == '__main__':
    block_cl = Conv3DResNetBlock
    block = block_cl(
        in_channels=10,
        out_channels=64,
        style='bottleneck'
    )
    num_params = sum([p.numel() for p in block.parameters() if p.requires_grad])
    print(num_params)

    block = block_cl(
        in_channels=10,
        out_channels=64,
        style='basic'
    )
    num_params = sum([p.numel() for p in block.parameters() if p.requires_grad])
    print(num_params)


    block = Conv3DInceptionBlock(
        in_channels=10,
        out_channels=64, 
    )
    num_params = sum([p.numel() for p in block.parameters() if p.requires_grad])
    print(num_params)

    block = Conv3DBlock(
        in_channels=10,
        out_channels=64,
        kernel_sizes=3, 
        strides=1,  
    )
    num_params = sum([p.numel() for p in block.parameters() if p.requires_grad])
    print('3D conv block', num_params)

    block = Conv3DResNetUpsampleBlock(
        in_channels=10,
        out_channels=64, 
        scale_factor=2,
        style='bottleneck'
    )
    num_params = sum([p.numel() for p in block.parameters() if p.requires_grad])
    print('upsample bottleneck', num_params)

    block = Conv3DResNetUpsampleBlock(
        in_channels=10,
        out_channels=64, 
        scale_factor=2,
        style='basic'
    )
    num_params = sum([p.numel() for p in block.parameters() if p.requires_grad])
    print('upsample basic', num_params)


    block = Conv3DInceptionBlockUpsampleBlock(
        in_channels=10,
        out_channels=64, 
        scale_factor=2,
    )
    num_params = sum([p.numel() for p in block.parameters() if p.requires_grad])
    print('upsample Inception', num_params)

    # from arm.c2farm.networks import Qattention3DNetWithContext
    # qnet = Qattention3DNetWithContext(
    #     in_channels=10,
    #     out_channels=1,
    #     voxel_size=16,
    #     out_dense=0,
    #     kernels=64,
    #     norm=None, 
    #     dense_feats=128,
    #     activation='lrelu',
    #     low_dim_size=10
    #     )




