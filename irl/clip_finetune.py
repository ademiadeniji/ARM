from email.policy import default
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import clip
from PIL import Image
import requests
from glob import glob
from natsort import natsorted
import torch.nn as nn 
import copy
import os 
import random 
import wandb
from torch.optim import Adam
from collections import defaultdict
from einops import rearrange, repeat 

import pickle 
import hydra 
from omegaconf import DictConfig, OmegaConf
from os.path import join

from dataset import ReplayDataset
from models import RewardMLP

def load_expert_trajs(task='push_button'):
    """Returned frames must be same as other replay data """
    all_trajs = []
    for var in range(18):
        for eps in range(20):
            path = f'/home/mandi/all_rlbench_data/{task}/variation{var}/episodes/episode{eps}/front_rgb/*'
            frames = natsorted(glob(path)) 
            all_trajs.append([Image.open(p) for p in frames])
    print('Scanned thru all expert trajs, found {}'.format(len(all_trajs)))
    return all_trajs

def load_dataset(cfg):
    print('Loading dataset from iteration {}'.format(cfg.dataset.idxs))
    all_success, all_fail = [], []
    all_data = defaultdict(list)
    for dataset_itr in sorted(cfg.dataset.idxs):
        replay_path = join(cfg.data_path, f'{cfg.task}/iteration{dataset_itr}')
        assert os.path.exists(replay_path), f'{replay_path} does not exist'

        for level in cfg.dataset.levels:
            if level == 'expert':
                continue
            paths = natsorted(glob(f'{replay_path}/{level}/episode*'))
            for path in paths:
                steps = natsorted(glob(f'{path}/*.pkl'))
                if len(steps) < cfg.dataset.nframes:
                    continue
                episode = []
                for i, step in enumerate(steps):
                    with open(step, 'rb') as f:
                        transition = pickle.load(f)
                    obs = transition.observation['front_rgb']
                    if obs.shape[-1] != 3 and obs.shape[0] == 3:
                        obs = obs.transpose(1,2,0)
                    episode.append(obs)

                    if i == len(steps) - 1:
                        final_obs = transition.final_observation['front_rgb']
                        if final_obs.shape[-1] != 3 and final_obs.shape[0] == 3:
                            final_obs = final_obs.transpose(1,2,0)
                            obs = Image.fromarray(np.uint8(final_obs))
                            episode.append(obs)

                all_data[level].append(episode)
    for k, trajs in all_data.items():
        print("Data level: {}, loaded {} episodes, max/min traj length: {}/{}".format(
            k, len(trajs), max([len(t) for t in trajs]), min([len(t) for t in trajs])))
    if 'expert' in cfg.dataset.levels:
        all_data['expert'] = load_expert_trajs(cfg.task)
    return ReplayDataset(cfg, all_data)

@hydra.main(config_name='config_rew', config_path='/home/mandi/ARM/irl')
def main(cfg: DictConfig) -> None: 
    
    train_dataset = load_dataset(cfg)  

    # load CLIP and MLP models:
    device = cfg.device
    model_name = "ViT-L/14" 
    clip_model, preprocess = clip.load(model_name, device=device)
    prompts = cfg.prompts
    text = clip.tokenize(prompts).to(device)
    print('Using prompt', prompts)
    
    mlp = RewardMLP(clip_model=clip_model).to(device)

    if len(cfg.load_dir) > 0:
        print('Loading model from {}'.format(cfg.load_dir))
        path = join(cfg.data_patch, cfg.load_dir, f"{cfg.load_step}.pt")
        mlp.load_state_dict(torch.load(path))

    print('Fine-tuning parameter count:', sum(p.numel() for p in mlp.parameters() if p.requires_grad))
    optim = Adam(mlp.parameters(), lr=cfg.lr)
    os.makedirs(f'{cfg.model_path}/{cfg.run_name}', exist_ok=('burn' in cfg.run_name))
    
    if cfg.log_wb:
        run = wandb.init(project="ContextARM", name=cfg.run_name)
        cfg_dict = {
            k: v for k, v in cfg.items() if not 'dataset' in k
        }
        for key in ['dataset']:
            for sub_key in cfg[key].keys():
                cfg_dict[key+'/'+sub_key] = cfg[key][sub_key]
        run.config.update(cfg_dict)

    train_iter = iter(train_dataset)
    for step in range(cfg.itrs):
        batch = next(train_iter).to(device) # (num_levels=2, num_trajs, num_frames, 3, 128, 128
        n, b, f, c, h, w = batch.shape
        assert c == 3 # channel dim 
        low_logits = mlp(
            rearrange(batch[0], 'b f ... -> (b f) ... '),
            text, 
            scale_logits=cfg.scale_logits
            )
        low_logits = rearrange(low_logits,'(b f) ... -> b f ...',  b=b, f=f)
        low_logits = torch.sum(low_logits, dim=1)
        lower = torch.exp(low_logits)

        high_logits= mlp(
            rearrange(batch[1], 'b f ... -> (b f) ... '),
            text,
            scale_logits=cfg.scale_logits)
        high_logits = rearrange(high_logits, '(b f) ... -> b f ...', b=b, f=f)
        high_logits = torch.sum(high_logits, dim=1)
        higher = torch.exp(high_logits)

        loss = - torch.log( (higher / (higher + lower + 1e-8)) + 1e-8 )
        loss = torch.mean(loss)
        if torch.isnan(loss):
            print('Warning! got nan loss at step', step) 
            continue
        acc = torch.sum(low_logits < high_logits) / b

        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % cfg.log_freq == 0:
            
            tolog = {'loss': loss.item(),
                'acc': acc.item(),
                'Train Step': step,
                'logits min': low_logits.min().item(), 
                'logits max': high_logits.max().item(),
                'logits mean': high_logits.mean().item(),
                }
           
            if cfg.log_wb:
                wandb.log(tolog)
            print(tolog)
            torch.save(mlp.state_dict(), f'{cfg.model_path}/{cfg.run_name}/{step}.pt')
 

if __name__ == "__main__":
    main()