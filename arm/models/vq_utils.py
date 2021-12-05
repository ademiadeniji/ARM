import torch 
import torch.nn as nn 
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
from omegaconf import OmegaConf
from arm.models.slowfast import TempResNet

import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange 

class Codebook(nn.Module):
    def __init__(self, training=True, embedding_size=2048, n_codes=5):
        super().__init__() 
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_size))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_size = embedding_size
        self._need_init = True
        self.training = True 

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x
    
    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        # flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        flat_inputs = rearrange(z, 'b c t h w -> (b t h w) c')
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z, update_codebook=True):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = rearrange(z, 'b c t h w -> (b t h w) c')
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        # shapes: '(b t h w) n_codes'  = '(b t h w) 1' - '(b t h w) n_codes' + '1 n_codes'

        encoding_indices = torch.argmin(distances, dim=1) # '(b t h w)'
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = rearrange(embeddings, 'b c t h w -> b w c t h')

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if update_codebook:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

if __name__ == '__main__':
    # test run a model
    model_cfg = OmegaConf.load('/home/mandi/ARM/conf/encoder/SlowRes.yaml')
    slow_18 = TempResNet(model_cfg)
    inp = torch.ones((4, 6, 3, 2, 128, 128)) # 
    b, k, ch, t, h, w = inp.shape

    out = slow_18.debug_forward( rearrange(inp, 'b k ch t h w -> (b k) ch t h w') )
    
    emb = rearrange(out, 'bk d -> bk d 1 1 1') # [b*k, 64] 
    #emb = slow_18._conv_out
    print(emb[0].eq(emb[1]))
    codebook = Codebook(embedding_size=64)
    # emb = slow_18._conv_out # [b*k, 2048, 2, 4, 4])
    # codebook = Codebook(embedding_size=2048)
    
    #encodings = rearrange( codebook(emb)['encodings'], '(b k) t h w -> b k (t h w)', b=b, k=k)
    print(codebook(emb)['encodings'].shape)
     


# embeddings torch.Size([1, 2048, 2, 4, 4])                                                                                                                
# encodings torch.Size([1, 2, 4, 4])                                                                                                                       
# commitment_loss torch.Size([])                                                                                                                           
# perplexity torch.Size([])     