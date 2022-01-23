import copy
import os
from typing import List
import torch
import torch.nn as nn
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, Summary
import numpy as np
import math
from abc import ABC, abstractmethod
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

NAME = 'ContextEmbedderAgent'
CONTEXT_KEY = 'demo_sample'

def _product_of_gaussians(mus, sigmas_squared):
        '''
        compute mu, sigma of product of gaussians
        '''
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
        return mu, sigma_squared
        
def _compute_kl_loss(z_means, z_vars):
        """ compute KL( q(z|c) || r(z) ) """
        latent_dim = z_means.shape[-1]
        device = z_means.device
        assert z_vars.shape[-1] == latent_dim
        prior = torch.distributions.Normal(
            torch.zeros(latent_dim).to(device), 
            torch.ones(latent_dim).to(device)
            )
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) \
            for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

class SequenceStrategy(ABC):

    @abstractmethod
    def apply(self, x):
        pass


class StackOnChannel(SequenceStrategy):

    def apply(self, x):
        # expect (B, T, C, ...)
        self._channels = x.shape[2]
        x = torch.split(x, 1, dim=1)
        return torch.cat(x, dim=2).squeeze(1)

    def inv(self, x):
        # expect (B, T * C, ...)
        return torch.cat(x.unsqueeze(1).split(self._channels, dim=2), dim=1)

 
class StackOnBatch(SequenceStrategy):

    def __init__(self):
        pass
        # self._timesteps = timesteps

    def apply(self, x):
        # expect (B, T, C, ...)
        bs, t, c = x.shape[:3]
        rest = x.shape[3:]
        return x.view(bs * t, c, *rest)
        # x = torch.split(x, 1, dim=1)
        # return torch.cat(x, dim=0).squeeze(1)


class ContextAgent(Agent):
    """Merge train and test time embeddingAgent 
    (from QAttentionMultitask.EmbeddingAgent) into one
    Warning for PEARL agent:
        - Context requires roughly "on-policy" transitions; 
        - need to compute product of gaussians; 
        - embedding net gets updated by both KL-loss and Qattention loss  
    
    """

    def __init__(self,
                 embedding_net: nn.Module,
                 camera_names: list,
                 with_action_context: bool,
                 is_train: bool,
                 # for traintime:
                 embedding_size: int,  
                 query_ratio: float,
                 margin: float = 0.1,
                 emb_lambda: float = 1.0,
                 save_context: bool = False, 
                 loss_mode: str = 'hinge', 
                 prod_of_gaus_factors_over_batch: bool = False, # for PEARL 
                 encoder_cfg: DictConfig = None,
                 one_hot: bool = False,
                 replay_update: bool = True, 
                 single_embedding_replay: bool = True, 
                 tau: float = 0.01,
                 param_update_freq: int = 10,
                 hidden_dim: int = -1,
                 replay_update_freq: int = -1,
                 use_target_embedder: bool = False,
                 ):
        self._embedding_net = embedding_net
        self._encoder_cfg = encoder_cfg
        self._camera_names = camera_names
        self._with_action_context = with_action_context
        self._sequence_strategy = StackOnChannel() # TODO(Mandi): other options here? 
        self._is_train = is_train
        # train time:
        if with_action_context and not isinstance(self._sequence_strategy, StackOnChannel):
            raise Exception('Embedding agent action context is only valid for StackOnChannel sequence strategy')
        self._current_context = None
        self._loss_mode = loss_mode
 
        #   TecNet: 
        self._query_ratio = query_ratio 
        self._margin = margin
        self._emb_lambda = emb_lambda
        #   PEARL
        if 'pearl' in self._loss_mode:
            self.pearl_latent_size = self._embedding_net.latent_size
        self._name = NAME 
        self._replay_update_freq = replay_update_freq # else, freeze emb net update 
        self._use_target_embedder = use_target_embedder
        self.param_update_freq = param_update_freq # use either for infoNCE loss or target embedding net 

        if use_target_embedder:
            self._embedding_net_target = copy.deepcopy(self._embedding_net)
            for p in self._embedding_net_target.parameters():
                p.requires_grad = False
            

        if 'info' in self._loss_mode:
            logging.info('Using contrastive infoNCE loss!')
            self._target_embedding_net = copy.deepcopy(self._embedding_net)
            
            emb_dim = embedding_size * 4 # hack 
            self._hidden_dim = hidden_dim if hidden_dim > 0 else emb_dim
            if hidden_dim > 0: 
                self._predictor = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
                )
            else:
                self._predictor = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim) # careful trying w/ and w/o layernorm 
                )
            self._predictor_target = copy.deepcopy(self._predictor)

            # self._W = nn.Parameter(torch.rand(emb_dim, emb_dim, requires_grad=True)) 
            self._emb_dim = emb_dim
            for p in chain(self._target_embedding_net.parameters(), self._predictor_target.parameters()):
                p.requires_grad = False  
            self.tau = tau 
       

        self._val_loss = None 
        self._val_embedding_accuracy = None 
        self._one_hot = one_hot 
        self._replay_update = replay_update
        self._replay_summaries, self._context_summaries = {}, {}
        self.single_embedding_replay = single_embedding_replay
        logging.info(f'Creating context agent that takes {query_ratio} of the total K samples as query')

    def build(self, training: bool, device: torch.device = None):
        """Train and Test time use the same build() """
        if device is None:
            device = torch.device('cpu') 
        self._embedding_net.set_device(device)
        #self._embedding_net.build()
        #self._embedding_net.to(device).train(training)
        self._device = device
        self._zero = torch.tensor(0.0, device=device)
        # use a separate optimizer here to update the params with metric loss,
        # optionally, qattention agents also have optimizers that update the embedding params here
        additional_params = []
        if 'info' in self._loss_mode: 
            self._target_embedding_net.set_device(device)
            self._predictor.to(device)
            self._predictor_target.to(device)
            self._W = nn.Parameter(torch.rand(self._hidden_dim, self._emb_dim, requires_grad=True, device=device)) 
            additional_params = [self._W] + [p for p in self._predictor.parameters() if p.requires_grad]
         
        if training:
            if 'hinge' in self._loss_mode:
                self._optimizer, self._optim_params = make_optimizer(
                    self._embedding_net, self._encoder_cfg, return_params=True, additional_params=additional_params)
            elif 'pearl' in self._loss_mode:
                self._optimizer = None 
                self._optim_params = [{'params': self._embedding_net.parameters()}]

        if self._use_target_embedder:
            self._embedding_net_target.set_device(device)
  

    def pearl_act_for_replay(self, step, replay_sample, output_loss=False):
        task_ids = replay_sample[TASK_ID] # this should be (B, K_action, ...)
        k_action = task_ids.shape[1]

        b, k, n, c, h, w = replay_sample['context_obs'].shape
        assert c == 3, 'channel dim mut at index 3'
        replay_sample['context_obs'] = rearrange(
            replay_sample['context_obs'], 'b k n c h w -> (b k) c n h w')
        for key in ['context_action', 'context_reward']:
            replay_sample[key] = rearrange(replay_sample[key], 'b k ... -> (b k) ...')
        out = self._embedding_net(replay_sample)
        out = rearrange(out, '(b k) d -> b k d', b=b)
        size = self.pearl_latent_size
        mus, sigmas = out[:, :, :size], F.softplus(out[:, :, size:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mus), torch.unbind(sigmas))]
        z_means = torch.stack([p[0] for p in z_params])
        z_vars = torch.stack([p[1] for p in z_params])

        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) \
                for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        z = [d.rsample() for d in posteriors]
        z = torch.stack(z)
        kl_loss = _compute_kl_loss(z_means, z_vars)

        action_embeddings = repeat(z, 'b d -> b k d', b=b, k=k_action) 
        info = {'emb_loss': kl_loss}
        self._replay_summaries = {
                    'replay_batch/'+k: torch.mean(v) for k,v in info.items()}
        return ActResult(action_embeddings, info=info)
 
    def act_for_replay(self, step, replay_sample, output_loss=False):
        """Use this to embed context only for qattention agent update"""
        if self._loss_mode == 'pearl':
            return self.pearl_act_for_replay(step, replay_sample, output_loss)

        data = replay_sample[CONTEXT_KEY].to(self._device) 
        if self._one_hot:
            return ActResult(data) 
        # NOTE(10/08) now replay sample is also shape (bsize, num_samples, vid_len, 3, 128, 128) 
        # note shape here is (bsize, video_len, 3, 128, 128), preprocess_agent squeezed the task dimension 
        b, k, n, ch, img_h, img_w = data.shape

        task_ids = replay_sample[TASK_ID] # this should be (B, K_action, ...)
        assert task_ids.shape[0] == b, f'B dimension in replay samples should all be the same, got {task_ids.shape[0]} and {b}'
        k_action = task_ids.shape[1]

        model_inp = rearrange(data, 'b k n ch h w -> (b k) ch n h w')
        embeddings = self._embedding_net(model_inp) # shape (b, embed_dim)
        embeddings = rearrange(embeddings, '(b k) d -> b k d', b=b, k=k)
        if self.single_embedding_replay:
            action_embeddings = repeat(embeddings[:, 0, :], 'b d -> b k d', b=b, k=k_action) 
        else:
            # following tecnet, control uses mean from support emb.!
            num_query = max(1, int(self._query_ratio * k)) # ugly hack cuz not enough validation data 
            num_support = int(k - num_query)
            _, support_embeddings = embeddings.split([num_query, num_support], dim=1)
            action_embeddings = repeat(
                support_embeddings.mean(dim=1), 'b d -> b k d', b=b, k=k_action)
        if 'info' not in self._loss_mode:
            action_embeddings = action_embeddings / action_embeddings.norm(dim=2, p=2, keepdim=True)
        # print('shapes of action embeddings vs embeddings:', action_embeddings.shape, embeddings.shape)
        act_result = ActResult(action_embeddings, info={})
        if self._replay_update and output_loss: 
            # self._optimizer.zero_grad()
            if 'hinge' in self._loss_mode:
                update_dict = self._compute_hinge_loss(embeddings, val=False)  
            elif 'info' in self._loss_mode:
                embeddings_target = rearrange(
                    self._target_embedding_net(model_inp),  '(b k) d -> b k d', b=b, k=k)
                update_dict = self._compute_info_loss(embeddings, embeddings_target, val=False)  
                if step % self.param_update_freq == 0:
                    self.soft_param_update()
            else:
                raise NotImplementedError
            emb_loss = update_dict['emb_loss']
            if step % self._replay_update_freq != 0:
                emb_loss *= 0

            info = {'emb_loss': emb_loss }
            if self._use_target_embedder:
                embeddings_target = rearrange(self._embedding_net_target(model_inp),  '(b k) d -> b k d', b=b, k=k)
                info['embeddings_target'] = embeddings_target 
                if step % self.param_update_freq == 0: # hard update 
                    for param, target_param in zip( self._embedding_net.parameters(), self._embedding_net_target.parameters()):
                        target_param.data.copy_(param.data)
            
            act_result = ActResult(
                action_embeddings, info=info)
            self._replay_summaries = {
                    'replay_batch/'+k: torch.mean(v) for k,v in update_dict.items()}
            
        return act_result
        
    def set_context_z(self, context_batch):
        assert 'pearl' in self._loss_mode
        size = self.pearl_latent_size
        
        if len(context_batch) == 0:
            prior = torch.distributions.Normal(
                torch.zeros(size), torch.ones(size)) 
            self.z = prior.rsample() 
            return 
        k, n, c, h, w = context_batch['context_obs'].shape
        assert c == 3, 'channel dim mut at index 3'
        context_batch['context_obs'] = rearrange(context_batch['context_obs'], 'k n c h w -> k c n h w')
        out = self._embedding_net(context_batch) # shape K, d, no batch dim 
         
        mu, sigma = out[:, :size], F.softplus(out[:, size:])
        z_params = _product_of_gaussians(mu, sigma)
        self.z_mean = z_params[0]
        self.z_var =  z_params[1]

        posterior = torch.distributions.Normal(
            self.z_mean, torch.sqrt(self.z_var))
         
        self.z = posterior.rsample().detach()
         

    
    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        """observation batch may require different input preprocessing, handle here """
        # print('context agent input:', observation.keys()) 
        if 'pearl' in self._loss_mode:
            return ActResult(self.z)
 
        data = observation[CONTEXT_KEY].to(self._device)
        if self._one_hot:
            return ActResult(data)
        
        k, n, ch, h, w = data.shape 
        num_query = max(1, int(self._query_ratio * k)) # ugly hack cuz not enough validation data 
        num_support = int(k - num_query)
        _, model_inp = data.split([num_query, num_support], dim=0) # no batch dim here!
        #print(data.shape) NOTE(1117) should change to shape (k, n, ch, img_h, img_w), doesn't have batch dim. 
        if self.single_embedding_replay:
            model_inp = model_inp[0:1,:]
        model_inp = rearrange(model_inp, 'k n ch h w -> k ch n h w')
        embeddings = self._embedding_net(model_inp).mean(dim=0, keepdim=True) # should be (1,d)
        if 'info' not in self._loss_mode:
            embeddings = embeddings / embeddings.norm(dim=1, p=2, keepdim=True) 
        self._current_context = embeddings.detach().requires_grad_(False)
    
        return ActResult(self._current_context)
        
    def set_new_context(self, observation: dict):
        with torch.no_grad():
            observations = []
            for n in self._camera_names:
                ob = observation[n]
                k, t, c, h, w = ob.shape
                ob_seq = self._sequence_strategy.apply(ob)
                if self._with_action_context:
                    action = self._sequence_strategy.apply(
                        observation['action'].type(torch.float32))
                    action_tiled = torch.reshape(action, (k, -1, 1, 1)).repeat(
                        1, 1, h, w)
                    ob_seq = torch.cat((ob_seq, action_tiled), -3)
                observations.append(ob_seq)
            embeddings = self._embedding_net(observations)
            embeddings = embeddings.view(1, k, -1)
            embeddings_norm = embeddings / embeddings.norm(
                dim=2, p=2, keepdim=True)
            context = embeddings_norm.mean(1)  # (B, E)
            self._current_context = (context / context.norm(
                dim=1, p=2, keepdim=True))[0]  # gives (E,)
        return self._current_context

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        device = self._device
        self._embedding_net.load_state_dict(
            torch.load(os.path.join(savedir, 'embedding_net.pt'), map_location=device)
            )
        if 'info' in self._loss_mode:
            self._target_embedding_net.load_state_dict(
            torch.load(os.path.join(savedir, 'target_embedding_net.pt'), map_location=device)
            )
            self._predictor.load_state_dict(
                torch.load(os.path.join(savedir, 'emb_predictor.pt'), map_location=device)
            )
            self._predictor_target.load_state_dict(
                torch.load(os.path.join(savedir, 'emb_predictor_target.pt'), map_location=device)
            )
            self._W = torch.load(os.path.join(savedir, 'emb_W.pt')).to(device)
            
    def save_weights(self, savedir: str):
        torch.save(
            self._embedding_net.state_dict(),
            os.path.join(savedir, 'embedding_net.pt'))
        if 'info' in self._loss_mode:
            torch.save(
                self._target_embedding_net.state_dict(),
                os.path.join(savedir, 'target_embedding_net.pt'))
            torch.save(
                self._predictor.state_dict(),
                os.path.join(savedir, 'emb_predictor.pt'))
            torch.save(
                self._predictor_target.state_dict(),
                os.path.join(savedir, 'emb_predictor_target.pt'))
            torch.save(self._W, os.path.join(savedir, 'emb_W.pt'))

    def soft_param_update(self): 
        tau = self.tau
        for param, target_param in zip( self._embedding_net.parameters(), self._target_embedding_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip( self._predictor.parameters(), self._predictor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        return

    def update_summaries(self) -> List[Summary]:
        if self._is_train:
            return self.update_train_summaries() 
        else:
            return self.update_test_summaries()
    
    def _preprocess_inputs(self, batch):
        #for collate_id, d in batch.items():
            #print('context agent update:', collate_id, d.get('front_rgb').shape)

        data = torch.stack([ d.get('front_rgb') for collate_id, d in batch.items()] ) 
        #print('context agent update:', data.shape)
        assert len(data.shape) == 6, 'Must be shape (b, num_episodes, num_frames, channels, img_h, img_w)'
        b, k, n, ch, img_h, img_w = data.shape 
        model_inp = rearrange(data, 'b k n ch h w -> (b k) ch n h w').to(self._device)
        return model_inp, b, k 

    def update(self, step, context_batch, val=False):
        # this is kept separate from replay_sample batch, s.t. we can contruct the
        # batch for embedding loss with more freedom 
        # data = torch.stack([ d.get('front_rgb') for collate_id, d in context_batch.items()] ) 
        # data = rearrange(data, 'b k n ch h w -> (b k) n ch h w')
        # temp_batch = {
        #    'demo_sample' : data
        # }
        # utils.visualize_batch(temp_batch, filename='/home/mandi/ARM/debug/ctxt_batch', img_size=128)
        # raise ValueError
        if self._one_hot:
            return {}
        model_inp, b, k = self._preprocess_inputs(context_batch)
         
        embeddings = self._embedding_net(model_inp)
        self._mean_embedding = embeddings.mean()
        embeddings = rearrange(embeddings, '(b k) d -> b k d', b=b, k=k)

        if not val:
            self._optimizer.zero_grad()
         
        if 'hinge' in self._loss_mode:
            update_dict = self._compute_hinge_loss(embeddings, val=val) 
        else:
            raise NotImplementedError
        self._context_summaries.update({
                f"context_batch/{'val' if val else 'train'}/"+k: torch.mean(v) for k,v in update_dict.items()}) 

        if not val:
            self._optimizer.zero_grad() 
            loss = update_dict['mean_emb_loss']
            loss.backward()
            self._optimizer.step()

        return update_dict

    def validate_context(self, step, context_batch):
        with torch.no_grad():
            val_dict = self.update(step, context_batch, val=True)
        return val_dict 

    
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
        #print(similarities.shape, 'previous version shape')
        if 'hinge-v2' in self._loss_mode: 
            query_embeddings, _ = embeddings_norm.split([num_query, num_support], dim=1) 
            dists = torch.cdist(query_embeddings.contiguous(), \
                    repeat(support_context, 'b d -> b b2 d', b2=b) ) # should give b, num_query, b
            #print(dists.shape)
            #assert  dists.transpose(1,2).shape == similarities.shape 
            similarities = dists.transpose(1,2)
        
        # Gets the diagonal to give (batch, query)
        diag = torch.eye(b, device=self._device)
        positives = torch.masked_select(similarities, diag.unsqueeze(-1).bool())  # (B * query)
        positives = positives.view(b, 1, num_query)  # (B, 1, query)

        negatives = torch.masked_select(similarities, diag.unsqueeze(-1) == 0)
        # (batch, batch-1, query)
        #print('shapes:', query_embeddings.shape, similarities.shape, diag.shape, )
        #print('negatives shape:', negatives.shape, positives.shape)
        negatives = negatives.view(b, b - 1, -1)

        loss = torch.max(self._zero, self._margin - positives + negatives)
        if val:
            self._val_loss = loss.mean() * self._emb_lambda
        else:
            self._loss = loss.mean() * self._emb_lambda

        # Summaries
        max_of_negs = negatives.max(1)[0]  # (batch, query)
        accuracy = positives[:, 0] > max_of_negs
        if val:
            self._val_embedding_accuracy = accuracy.float().mean() 
        else:
            self._embedding_accuracy = accuracy.float().mean()
        
        #print(support_context.shape, support_embeddings[:,0].shape)
        if 'hinge-v2' in self._loss_mode: 
            dists = torch.cdist(query_embeddings, repeat(support_embeddings[:,0], 'b d -> b b2 d', b2=b) ) # should give b, num_query, b
            # print(dists.shape)
            assert  dists.transpose(1,2).shape == similarities.shape 
            similarities = dists.transpose(1,2)
        else:
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
            'emb_loss': loss * self._emb_lambda,
            'mean_emb_loss': loss.mean() * self._emb_lambda,
            'emd_acc': accuracy.float().mean(),
            'emd_single_acc':  single_accuracy.float().mean(),
        }

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

        # if self._current_context is None:
        #     prefix = 'context/train'
        #     summaries.extend([
        #         ScalarSummary('%s_loss' % prefix, self._loss),
        #         ScalarSummary('%s_accuracy' % prefix, self._embedding_accuracy),
        #         ScalarSummary('%s_mean_embedding' % prefix, self._mean_embedding),
        #     ])
        #     if self._val_embedding_accuracy is not None:
        #         prefix = 'context/val'
        #         summaries.extend([
        #         ScalarSummary('%s_loss' % prefix, self._val_loss),
        #         ScalarSummary('%s_accuracy' % prefix, self._val_embedding_accuracy)
        #         ])
            # not logging parameters yet 
            # for tag, param in self._embedding_net.named_parameters():
            #     assert not torch.isnan(param.grad.abs() <= 1.0).all()
            #     summaries.append(
            #         HistogramSummary('%s/gradient/%s' % (prefix, tag), param.grad))
            #     summaries.append(
            #         HistogramSummary('%s/weight/%s' % (prefix, tag), param.data))
        return summaries
    
    def update_test_summaries(self) -> List[Summary]:
        return []

     