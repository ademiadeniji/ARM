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
import math
from arm.models.utils import make_optimizer # tie the optimizer definition closely with embedding nets 
from arm import utils
from omegaconf import DictConfig
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
 
CONTEXT_KEY = 'demo_sample'

  
class DiscreteContextAgent(Agent):
    """using different encode for now"""

    def __init__(self,
                 is_train: bool,   
                 loss_mode: str = 'dvae',  
                 one_hot: bool = False,
                 replay_update: bool = False, 
                 single_embedding_replay: bool = True,  
                 replay_update_freq: int = -1,
                 ):   
        self._is_train = is_train
        # train time:
        self._current_context = None
        self._loss_mode = loss_mode
        
        self._val_loss = None 
        self._val_embedding_accuracy = None 
        self._one_hot = one_hot 
        self._replay_update = replay_update
        self._replay_summaries, self._context_summaries = {}, {}
        self.single_embedding_replay = single_embedding_replay
        logging.info(f'Creating context agent with discrete embeddings')

    def build(self, training: bool, device: torch.device = None):
        """Train and Test time use the same build() """
        self._optim_params = []
        if device is None:
            device = torch.device('cpu') 
        if 'dvae' in self._loss_mode:
            self._embedding_net = load_model("/home/mandi/ARM/encoder.pkl", device)
        else:
            raise NotImplementedError 
        self._device = device 
        # use a separate optimizer here to update the params with metric loss,
        # optionally, qattention agents also have optimizers that update the embedding params here
        additional_params = []
        # if training:
        #     self._optimizer, self._optim_params = make_optimizer(
        #         self._embedding_net, self._encoder_cfg, return_params=True, additional_params=additional_params)
 
    def act_for_replay(self, step, replay_sample, output_loss=False):
        """Use this to embed context only for qattention agent update"""
        data = replay_sample[CONTEXT_KEY].to(self._device) 
        if self._one_hot:
            return ActResult(data) 
        # NOTE(10/08) now replay sample is also shape (bsize, num_samples, vid_len, 3, 128, 128) 
        # note shape here is (bsize, video_len, 3, 128, 128), preprocess_agent squeezed the task dimension 
        b, k, n, ch, img_h, img_w = data.shape

        task_ids = replay_sample[TASK_ID] # this should be (B, K_action, ...)
        assert task_ids.shape[0] == b, f'B dimension in replay samples should all be the same, got {task_ids.shape[0]} and {b}'
        k_action = task_ids.shape[1]

        if 'dvae' in self._loss_mode:
            assert n == 1, 'pre-trained dvae takes single images'
            model_inp = rearrange(data, 'b k n ch h w -> (b k n) ch h w')
            model_inp = map_pixels(model_inp)
            embeddings = self._embedding_net(model_inp) # shape (bk, K=8192, 16, 16)  
            embeddings = torch.argmax(embeddings, axis=1).float()
            embeddings = rearrange(embeddings, '(b k) h w -> b k (h w)', b=b, k=k)
            assert embeddings.shape[-1] == 256, 'should flatten into 16x16'

        if self.single_embedding_replay:
            action_embeddings = repeat(embeddings[:, 0, :], 'b d -> b k d', b=b, k=k_action) 
        else:
            raise NotImplementedError
        act_result = ActResult(action_embeddings, info={'emb_loss': torch.zeros(action_embeddings.shape)})
        if self._replay_update and output_loss: 
            # self._optimizer.zero_grad()
            raise NotImplementedError
            # emb_loss = update_dict['emb_loss']
            # if step % self._replay_update_freq != 0:
            #     emb_loss *= 0
            # act_result = ActResult(action_embeddings, info={'emb_loss': emb_loss })
            # self._replay_summaries = {
            #         'replay_batch/'+k: torch.mean(v) for k,v in update_dict.items()}
            
        return act_result
        
    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        """observation batch may require different input preprocessing, handle here """
        # print('context agent input:', observation.keys()) 
        data = observation[CONTEXT_KEY].to(self._device)
        if self._one_hot:
            return ActResult(data)
        
        k, n, ch, h, w = data.shape 
        if self.single_embedding_replay:
            data = data[0:1,:]
            k = 1
        else:
            raise NotImplementedError
        
        if 'dvae' in self._loss_mode:
            assert n == 1, 'pre-trained dvae takes single images'
            model_inp = rearrange(data, 'k n ch h w -> (k n) ch h w')
            model_inp = map_pixels(model_inp)
            
            embeddings = self._embedding_net(model_inp) # shape (bk, K=8192, 16, 16)  
            embeddings = torch.argmax(embeddings, axis=1).float()
            embeddings = rearrange(embeddings, 'kn h w -> kn (h w)')

         
        self._current_context = embeddings.detach().requires_grad_(False)
        return ActResult(self._current_context)
 
    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        device = self._device
        if 'dvae' in self._loss_mode:
            self._embedding_net = load_model("/home/mandi/DALL-E/encoder.pkl", device)
        else:
            raise NotImplementedError
            
    def save_weights(self, savedir: str):
        if 'dvae' in self._loss_mode:
            pass 
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


    def update_train_summaries(self) -> List[Summary]:
        summaries = []
        if self._one_hot:
            return summaries

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

     