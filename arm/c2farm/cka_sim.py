"""
credit: https://github.com/jayroxis/CKA-similarity/
"""
import math
import torch
import numpy as np
from  networks import Qattention3DNet
from einops import rearrange, repeat


class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K): 
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            if torch.max(KX) == 0:
                sigma = 1
            else:
                mdist = torch.median(KX[KX != 0])
                sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(
            self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma))
            )

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        print((hsic, var1, var2))
        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    cuda_cka = CudaCKA(device)

    # X = torch.randn(10000, 100, device=device)
    # Y = torch.randn(10000, 100, device=device)

    # print('Linear CKA, between X and Y: {}'.format(cuda_cka.linear_CKA(X, Y)))
    # print('Linear CKA, between X and X: {}'.format(cuda_cka.linear_CKA(X, X)))

    # print('RBF Kernel CKA, between X and Y: {}'.format(cuda_cka.kernel_CKA(X, Y)))
    # print('RBF Kernel CKA, between X and X: {}'.format(cuda_cka.kernel_CKA(X, X)))
    net = Qattention3DNet(
                in_channels=32 + 3 + 1 + 3,
                out_channels=1,
                voxel_size=16,
                out_dense=0,
                kernels=64,
                norm=None,
                dense_feats=128,
                activation='lrelu',
                low_dim_size=8
                )
    net.build() 
    tot = len( list(net.parameters()) )
    sim = 0
    c = 0
    for i, p in enumerate(net.parameters()):
        # print(p.shape)
        if len(p) > 0:
            p = repeat(p.data, '... -> (...) s1').to(device)
             
            sim += cuda_cka.linear_CKA(p, p)
            c += 1
            if i > 0 and i % 10 == 0:
                print('done comparing {}/{}'.format(i, sim/c))
   
    print('all done', sim/c)