from typing import List

import torch

from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary, ImageSummary
from einops import rearrange, reduce
import logging
from arm import utils 

class PreprocessAgent(Agent):

    def __init__(self,
                 pose_agent: Agent,
                 context_agent: Agent = None):
        self._pose_agent = pose_agent
        self._context_agent = context_agent 

    def build(self, training: bool, device: torch.device = None, context_device: torch.device = None):
        # NOTE build context agent first to get its optim paramters
        if self._context_agent is not None:
            context_device = context_device if context_device is not None else device 
            self._context_agent.build(training, context_device)
        
        self._pose_agent.build(training, device)
        self._device = device 
        self._replay_sample = None 
        
    def _norm_rgb_(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def visualize_batch(self, step, replay_sample):

        utils.visualize_batch(replay_sample, filename=f'/home/mandi/ARM/debug/replay_batch_step{step}', img_size=128)

    def update(self, step: int, replay_sample: dict) -> dict:
        # Samples are (N, K, ...) where we sample N buffers for each batch and get K transitions from each buffer
        #for k, v in replay_sample.items():
            #print('preprocess agent update:', k, v.shape, v[:,0].shape )
        # replay_sample = {k: v[:, 0] for k, v in replay_sample.items()}
        for k, v in replay_sample.items():
            if 'rgb' in k:
                replay_sample[k] = self._norm_rgb_(v)
        # self._replay_sample = replay_sample
        self._replay_sample = {k: rearrange(v, 'n k ... -> (n k) ...') for k, v in replay_sample.items()} # the stack agent does another reshape, here is just for logging purpose 
        for k, v in replay_sample.items():
            if isinstance(v, torch.Tensor):
                replay_sample[k] = v.to(self._device) 
        pose_dict = self._pose_agent.update(step, replay_sample)
         
        return pose_dict

    def update_context_via_qagent(self, step: int, replay_sample: dict, classify: bool, emb_weight: float):
        for k, v in replay_sample.items():
            if 'rgb' in k:
                replay_sample[k] = self._norm_rgb_(v) 
        self._replay_sample = {
            k: rearrange(v, 'n k ... -> (n k) ...') for k, v in replay_sample.items()} # the stack agent does another reshape, here is just for logging purpose 
        for k, v in replay_sample.items():
            if isinstance(v, torch.Tensor):
                replay_sample[k] = v.to(self._device) 
        return self._pose_agent.update_context_only(step, replay_sample, classify, emb_weight)

    def rebuild_optimizer(self)  -> None:
        agent = self._pose_agent._qattention_agents[0]
        q_params = agent._q.parameters()
        q_params = [{"params": q_params, 'lr': agent._lr}] 
        agent._optimizer = torch.optim.Adam(q_params, lr=agent._lr,
            weight_decay=agent._lambda_weight_l2)


    def update_context(self, step: int, context_batch: dict) -> dict:
        self._context_batch = context_batch 
        return self._context_agent.update(step, context_batch)
    
    def validate_context(self, step: int, context_batch: dict) -> dict:
        return self._context_agent.validate_context(step, context_batch)

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        
        observation = {
            k: v.clone().detach().to(self._device) if isinstance(v, torch.Tensor) else \
            torch.tensor(v).to(self._device)  for k, v in observation.items()}
        for k, v in observation.items():
            if 'rgb' in k:
                observation[k] = self._norm_rgb_(v)
        # if context is needed for every qattention, will also have access to context_agent to handle it
        act_res = self._pose_agent.act(step, observation, deterministic)
        act_res.replay_elements.update({'demo': False})
        return act_res

    def update_summaries(self) -> List[Summary]:
        prefix = 'inputs'
        demo_f = self._replay_sample['demo'].float()
        demo_proportion = demo_f.mean()
        tile = lambda x: torch.squeeze(
            torch.cat(x.split(1, dim=1), dim=-1), dim=1)
        sums = [
            ScalarSummary('%s/demo_proportion' % prefix, demo_proportion),
            HistogramSummary('%s/low_dim_state' % prefix,
                    self._replay_sample['low_dim_state']),
            HistogramSummary('%s/low_dim_state_tp1' % prefix,
                    self._replay_sample['low_dim_state_tp1']),
            ScalarSummary('%s/low_dim_state_mean' % prefix,
                    self._replay_sample['low_dim_state'].mean()),
            ScalarSummary('%s/low_dim_state_min' % prefix,
                    self._replay_sample['low_dim_state'].min()),
            ScalarSummary('%s/low_dim_state_max' % prefix,
                    self._replay_sample['low_dim_state'].max()),
            ScalarSummary('%s/timeouts' % prefix,
                    self._replay_sample['timeout'].float().mean()),
        ]
        fronts = []
        for k, v in self._replay_sample.items():
            # if 'rgb' in k or 'point_cloud' in k:
            #     if 'rgb' in k:
            #         # Convert back to 0 - 1
            #         v = (v + 1.0) / 2.0
            #     sums.append(ImageSummary('%s/%s' % (prefix, k), tile(v)))
                
            if 'front' in k  and  ('rgb' in k or 'point_cloud' in k ):
                v = (v + 1.0) / 2.0 if 'rgb' in k else v 
                fronts.append(tile(v)) # tile(v) has shape (b, 3, 128, 128)
        fronts = torch.cat(fronts, dim=3)
        sums.append(ImageSummary('%s/%s' % (prefix, 'front_cam_inputs'),  fronts))
         
        if 'sampling_probabilities' in self._replay_sample:
            sums.extend([
                HistogramSummary('replay/priority',
                                 self._replay_sample['sampling_probabilities']),
            ])
        sums.extend(self._pose_agent.update_summaries())
        if self._context_agent is not None:
            sums.extend(self._context_agent.update_summaries())
        
        # for s in sums:
        #     print(s.name, s.value.shape, type(s))
        return sums

    def act_summaries(self) -> List[Summary]: 
        return self._pose_agent.act_summaries() # context stuff should already be handled 

    def load_weights(self, savedir: str):
        self._pose_agent.load_weights(savedir)
        if self._context_agent is not None:
            self._context_agent.load_weights(savedir)


    def save_weights(self, savedir: str):
        self._pose_agent.save_weights(savedir)
        if self._context_agent is not None:
            self._context_agent.save_weights(savedir)

    def reset(self) -> None:
        self._pose_agent.reset()
        if self._context_agent is not None:
            self._context_agent.reset()

