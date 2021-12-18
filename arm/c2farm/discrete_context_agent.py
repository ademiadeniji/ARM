"""
using discrete embeddings: dVAE or VQ-VAE
for pre-trained dVAE: remember to set dataset.num_steps_per_episode=1 and dataset.defer_transforms
"""
import copy
import os
from typing import List
import torch
import torch.nn as nn
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, Summary
import numpy as np

from arm.models.utils import make_optimizer # tie the optimizer definition closely with embedding nets 
from arm import utils

from einops import rearrange, reduce, repeat, parse_shape
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from torch import einsum
from itertools import chain 
import logging
from yarr.utils.multitask_rollout_generator import TASK_ID, VAR_ID # use this to get K_action


import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dall_e  import map_pixels, unmap_pixels, load_model
from arm.models.vq_utils import Codebook
CONTEXT_KEY = 'demo_sample'
LRELU_SLOPE = 0.02 

def sample_gumbel_softmax(z, tau=0.01, eps=1e-20): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  if len(z.shape) == 3:
      z = rearrange(z, 'b k d -> b k 1 1 1 d')
  assert len(z.shape) == 6, 'should be shape B n h w d and feature is the last dim'
  d = z.shape[-1] 
  sampled_unif = torch.rand(z.shape)
  sampled_gumbel = -torch.log(-torch.log(sampled_unif + eps) + eps).to(z.device)
  y = z + sampled_gumbel
  y_continous = F.softmax( y / tau, dim=-1)
  # print(z.shape, y_continous.shape)
  y_discrete = F.one_hot( torch.argmax(y_continous, axis=-1), num_classes=d) 
  # print(y_discrete.shape)
  y_discrete = (y_discrete - y_continous).detach() + y_continous

  return y_continous, y_discrete

class DiscreteContextAgent(Agent):
    """using different encode for now"""

    def __init__(self, 
                 is_train: bool,   
                 loss_mode: str = 'dvae',  
                 one_hot: bool = False,
                 replay_update: bool = True, 
                 single_embedding_replay: bool = True,  
                 replay_update_freq: int = -1,
                 embedding_net: nn.Module = None,  
                 encoder_cfg: dict = None, 
                 latent_dim: int = 1, 
                 anneal_schedule: float = 5e-4,
                 temp_update_freq: int = 100,
                 query_ratio: float = 0.3,
                 margin: float = 0.1,
                 discrete_before_hinge: bool = False, 
                 dev_cfg: dict = {},
                 ):   
        self._is_train = is_train
        # train time:
        self._current_context = None
        self._loss_mode = loss_mode
        self._embedding_net = embedding_net
        self._encoder_cfg = encoder_cfg
        self._val_loss = None 
        self._val_embedding_accuracy = None 
        self._one_hot = one_hot 
        self._replay_update = replay_update
        self._replay_summaries, self._context_summaries = {}, {}
        self.single_embedding_replay = single_embedding_replay 
        self._latent_dim =  latent_dim
        self._anneal_schedule = anneal_schedule
        self._temp_update_freq = temp_update_freq
        self.tau = 1 
        self._query_ratio, self._margin = query_ratio, margin 
        self.discrete_before_hinge = discrete_before_hinge
        self._dev_cfg = dev_cfg 
        logging.info(f'Creating context agent with discrete embeddings')

    def build(self, training: bool, device: torch.device = None):
        """Train and Test time use the same build() """
        self._embedding_net.set_device(device)
        self._optim_params = []
        if device is None:
            device = torch.device('cpu') 
        
        if 'dvae' in self._loss_mode:
            self._embedding_net = load_model("/home/mandi/ARM/encoder.pkl", device)
            if self._dev_cfg.get('use_conv', False):
                self.conv3d = nn.Conv3d(
                    in_channels=8192, out_channels=self._dev_cfg.conv_out, 
                    kernel_size=tuple(self._dev_cfg.conv_kernel), 
                    stride=self._dev_cfg.stride
                    ).to(device)
                nn.init.kaiming_uniform_(self.conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
                nn.init.zeros_(self.conv3d.bias)
                self.activate = nn.LeakyReLU(negative_slope=LRELU_SLOPE)

                self._optim_params = [ {"params": self.conv3d.parameters() } ]
        elif 'vqvae' in self._loss_mode:
            self._codebook = Codebook() 

        elif 'gumbel' in self._loss_mode:
            additional_params = []
   
            if training:
                self._optimizer, self._optim_params = make_optimizer(
                self._embedding_net, self._encoder_cfg, return_params=True, additional_params=additional_params)

        else:
            raise NotImplementedError 
        self._device = device 
        # use a separate optimizer here to update the params with metric loss,
        # optionally, qattention agents also have optimizers that update the embedding params here
        
        
    def act_for_replay(self, step, replay_sample, output_loss=False):
        """Use this to embed context only for qattention agent update"""
        data = replay_sample[CONTEXT_KEY].to(self._device) 
        # if self._one_hot:
        #     return ActResult(data) 
        # NOTE(10/08) now replay sample is also shape (bsize, num_samples, vid_len, 3, 128, 128) 
        # note shape here is (bsize, video_len, 3, 128, 128), preprocess_agent squeezed the task dimension 
        b, k, n, ch, img_h, img_w = data.shape

        task_ids = replay_sample[TASK_ID] # this should be (B, K_action, ...)
        assert task_ids.shape[0] == b, f'B dimension in replay samples should all be the same, got {task_ids.shape[0]} and {b}'
        k_action = task_ids.shape[1]

        if 'dvae' in self._loss_mode:
            #assert n == 1 or n == 4, f'pre-trained dvae takes single images, not {n}' 
            model_inp = rearrange(data, 'b k n ch h w -> (b k n) ch h w')
            # if n == 4: # grid layout!
            #     upp = torch.cat([ data[:,:,0], data[:,:,1] ], dim=4)
            #     down = torch.cat([ data[:,:,2], data[:,:,3] ], dim=4)
            #     data = torch.cat([upp, down], dim=3) # b, k, 3, 256, 256 
            #     model_inp = rearrange(data, 'b k ch h w -> (b k) ch h w')
            model_inp = map_pixels(model_inp)
            embeddings = self._embedding_net(model_inp) # shape (bk, K=8192, 16, 16)
            
            z = torch.argmax(embeddings, axis=1) 
            embeddings = F.one_hot(z, num_classes=8192).float()
            if n == 1:
                embeddings = rearrange(embeddings, '(b k) h w d -> b k (h w d)', b=b, k=k) 
            else: # use OR
                if self._dev_cfg.get('use_conv', False):
                    embeddings = self.activate(
                        self.conv3d( rearrange(embeddings, '(bk n) h w d -> bk d n h w', bk=b*k, n=n) )
                     ) # -> b k d' 1 h' w'
                    
                    embeddings = rearrange(embeddings, '(b k) d n h w -> b k (d n h w)', b=b, k=k)
                else:
                    embeddings = rearrange(embeddings, '(b k n) h w d -> b k n (h w d)', b=b, k=k, n=n).sum(2)
                    embeddings = torch.min(torch.ones_like(embeddings).to(self._device), embeddings) # -> b k (d h w)
                
             
            action_embeddings = repeat(embeddings[:, 0, :], 'b d -> b k d', b=b, k=k_action)
            act_result = ActResult(action_embeddings, info={'emb_loss': torch.zeros(action_embeddings.shape)})

        elif 'gumbel' in self._loss_mode:
            model_inp = rearrange(data, 'b k n ch h w -> (b k) ch n h w')
            out = self._embedding_net(model_inp) # shape (bk, embed_dim)
            #print(out.shape)
            out = rearrange(out, '(b k) d -> b k d', b=b, k=k)
            z = out 
            if self._latent_dim == 3:
                conv_out = self._embedding_net._conv_out[0]
                z = rearrange(conv_out, '(b k) d n h w -> b k n h w d', b=b, k=k)
                # d=2048
                # z   = torch.argmax(out, axis=1) 
                # embeddings = F.one_hot(z, num_classes=2048).float() # (bk n h w 2048) 
            
            if step % self._temp_update_freq == 0:
                self.tau = max(0.05, np.exp(- self._anneal_schedule * step ))
                

            embeddings_cont, embeddings_discrete  = sample_gumbel_softmax(z, self.tau) # b k n h w d
            if self.discrete_before_hinge:
                out.detach()
                
                discrete_out = [rearrange(embeddings_discrete, 'b k n h w d -> (b k) d n h w')]
                out = self._embedding_net.head(discrete_out)
                out = rearrange(out, '(b k) d -> b k d', b=b, k=k)
 
            # just use one demo sample 
            action_embeddings = repeat(embeddings_discrete[:, 0, :], 'b n h w d -> b k (n h w d)', b=b, k=k_action)
            update_dict = self._compute_hinge_loss(out, val=False)  
            emb_loss = update_dict['emb_loss']
            
            act_result = ActResult(action_embeddings, info={'emb_loss': emb_loss })
            self._replay_summaries = {
                    'replay_batch/'+k: torch.mean(v) for k,v in update_dict.items()}
            self._replay_summaries['tau'] = self.tau 
 
        
        elif 'vqvae' in self._loss_mode:
            model_inp = rearrange(data, 'b k n ch h w -> (b k) ch n h w')
            x = self._embedding_net(model_inp) # shape (b, embed_dim)
            conv_out = self._embedding_net._conv_out[0]
            vq_outs = self.codebook(conv_out, update_codebook=(step % self._replay_update_freq != 0))
            embeddings = rearrange(vq_outs, '(b k) d -> b k d', b=b, k=k) 

        return act_result
        
    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        """observation batch may require different input preprocessing, handle here """
        # print('context agent input:', observation.keys()) 
        data = observation[CONTEXT_KEY].to(self._device)
 
        k, n, ch, h, w = data.shape 
        if self.single_embedding_replay:
            data = data[0:1,:]
            k = 1
          
        if 'dvae' in self._loss_mode:
            #assert n == 1 or n == 4, 'pre-trained dvae takes single images'
            # if n == 4: # grid layout!
            #     upp = torch.cat([data[:,0], data[:,1]], dim=3)
            #     down = torch.cat([data[:,2], data[:,3]], dim=3)
            #     model_inp = torch.cat([upp, down], dim=2) # 1, 3, 256, 256 
            # else:
            model_inp = rearrange(data, 'k n ch h w -> (k n) ch h w')
            model_inp = map_pixels(model_inp)
            embeddings = self._embedding_net(model_inp) # shape (bk, K=8192, 16, 16)  
             
            z = torch.argmax(embeddings, axis=1)
            embeddings = F.one_hot(z, num_classes=8192).float()
            if n > 1:
                if self._dev_cfg.get('use_conv', False):
                    embeddings = self.activate(
                        self.conv3d( rearrange(embeddings, '(k n) h w d -> k d n h w', k=k, n=n) ) 
                        )# -> b k d' 1 h' w'
                    embeddings = rearrange(embeddings, 'k d n h w -> k (d n h w)')
                else:
                    embeddings = rearrange(embeddings, '(k n) h w d -> k n (h w d)', k=k, n=n).sum(1)
                    embeddings = torch.min(torch.ones_like(embeddings).to(self._device), embeddings)
            else:
                embeddings = rearrange(embeddings, 'kn h w d -> kn (h w d)')
             
              
            embeddings = embeddings.mean(dim=0, keepdim=True) # k=1 anyway

        elif 'gumbel' in self._loss_mode:
            model_inp = rearrange(data[0:1, :],  'k n ch h w -> k ch n h w' )# just take sample 
            out = self._embedding_net(model_inp) # shape (1, embed_dim) 
            z = out 
            num_classes = out.shape[-1]
            if self._latent_dim == 3:
                z = self._embedding_net._conv_out[0] #'1 d n h w - 
                num_classes = 2048
                # d=2048
            
            z   = torch.argmax(z, axis=1) 
            embeddings = rearrange(
                F.one_hot(z, num_classes=num_classes).float(), '1 ... -> 1 (...)') # (1 (n h w 2048)) or (1 d)
      
        self._current_context = embeddings.detach().requires_grad_(False)
        return ActResult(self._current_context)
 
    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        device = self._device
        if 'dvae' in self._loss_mode:
            self._embedding_net = load_model("/home/mandi/ARM/encoder.pkl", device)
        elif 'gumbel' in self._loss_mode:
            self._embedding_net.load_state_dict(
                torch.load(os.path.join(savedir, 'embedding_net.pt'), map_location=device)
                )
        else:
            raise NotImplementedError
            
    def save_weights(self, savedir: str):
        if 'dvae' in self._loss_mode:
            pass 
        elif 'gumbel' in self._loss_mode:
            torch.save(
                self._embedding_net.state_dict(),
                os.path.join(savedir, 'embedding_net.pt'))
        else:
            raise NotImplementedError

    def update(self, step):
        return {}

    def update_summaries(self) -> List[Summary]:
        if self._is_train:
            return self.update_train_summaries() 
        else:
            return self.update_test_summaries()
     
 
    def validate_context(self, step, context_batch):
        with torch.no_grad():
            val_dict = self.update(step, context_batch, val=True)
        return val_dict 
 
    def _compute_info_loss(self, embeddings, embeddings_target, val=False):
        b, k, d = embeddings.shape 
        z_a     = self._predictor(rearrange(embeddings, 'b k d -> (b k) d'))
        shuffled_idx = torch.randperm(k)
        z_pos   = rearrange( embeddings_target[:, shuffled_idx], 'b k d -> (b k) d').detach()
        #print(z_a.device, self._W.device, z_pos.device)
        logits = einsum('af,fd,bd->ab', z_a, self._W, z_pos) # shape (Bk, Bk)
        logits = logits - reduce(logits, 'a b -> a 1', 'max')
        labels = torch.arange(logits.shape[0]).long().to(logits.get_device())
        info_loss = F.cross_entropy(logits, labels, reduction='none')
        
        return { 
            'emb_loss': rearrange(info_loss, '(b k) -> b k', b=b, k=k), 
            'mean_emb_loss': info_loss.mean(),
        }

    def _compute_hinge_loss(self, embeddings, val=False): 
        b, k, d = embeddings.shape 
        embeddings_norm = embeddings / embeddings.norm(dim=2, p=2, keepdim=True)

        num_query = 1 if val else max(1, int(self._query_ratio * k)) # ugly hack cuz not enough validation data 
        num_support = int(k - num_query)
 
        # support_embeddings = embeddings_norm[:, num_support:]
        # query_embeddings = embeddings_norm[:, :num_query].reshape(b * num_query, -1) 
        query_embeddings, support_embeddings = embeddings_norm.split([num_query, num_support], dim=1)
        query_embeddings = query_embeddings.reshape(b * num_query, -1) 
        # norm query too?
        # print('context agent embedding size and val?: ', embeddings_norm.shape, val )
        support_context = support_embeddings.mean(1)  # (B, E)
        support_context = support_context / support_context.norm(dim=1, p=2, keepdim=True) # B, d
        similarities = support_context.matmul(query_embeddings.transpose(0, 1))
        similarities = similarities.view(b, b, num_query)  # (B, B, queries)
         
        # Gets the diagonal to give (batch, query)
        diag = torch.eye(b, device=self._device)
        positives = torch.masked_select(similarities, diag.unsqueeze(-1).bool())  # (B * query)
        positives = positives.view(b, 1, num_query)  # (B, 1, query)

        negatives = torch.masked_select(similarities, diag.unsqueeze(-1) == 0)
        # (batch, batch-1, query)
        #print('shapes:', query_embeddings.shape, similarities.shape, diag.shape, )
        #print('negatives shape:', negatives.shape, positives.shape)
        negatives = negatives.view(b, b - 1, -1)

        loss = torch.max(torch.zeros_like(negatives).to(self._device), self._margin - positives + negatives)
        if val:
            self._val_loss = loss.mean()  
        else:
            self._loss = loss.mean()  

        # Summaries
        max_of_negs = negatives.max(1)[0]  # (batch, query)
        accuracy = positives[:, 0] > max_of_negs
        if val:
            self._val_embedding_accuracy = accuracy.float().mean() 
        else:
            self._embedding_accuracy = accuracy.float().mean() 
         
        similarities = support_embeddings[:,0].matmul(query_embeddings.transpose(0, 1))
        similarities = similarities.view(b, b, num_query)

        positives = torch.masked_select(similarities, diag.unsqueeze(-1).bool())  # (B * query)
        positives = positives.view(b, 1, num_query)  # (B, 1, query) 
        negatives = torch.masked_select(similarities, diag.unsqueeze(-1) == 0) 
        negatives = negatives.view(b, b - 1, -1)
        max_of_negs = negatives.max(1)[0]  # (batch, query)
        single_accuracy = positives[:, 0] > max_of_negs
        return {
            # 'context': support_context,
            'emb_loss': loss,
            'mean_emb_loss': loss.mean(),
            'emd_acc': accuracy.float().mean(),
            'emd_single_acc':  single_accuracy.float().mean(),
        }
   


    def update_train_summaries(self) -> List[Summary]:
        summaries = []
        # if self._one_hot:
        #     return summaries

        prefix = 'ContextAgent'
        if self._replay_summaries is not None:
            summaries.extend([
                ScalarSummary(f'{prefix}/{key}', v) for key, v in self._replay_summaries.items()
                ])
        if self._context_summaries is not None:
            summaries.extend([
                ScalarSummary(f'{prefix}/{key}', v) for key, v in self._context_summaries.items()
                ])
 
        return summaries
    
    def update_test_summaries(self) -> List[Summary]:
        return []

     