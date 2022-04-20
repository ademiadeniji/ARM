from email.policy import default
from threading import activeCount
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

    if 'expert' in cfg.dataset.levels:
        all_data['expert'] = load_expert_trajs(cfg.task)

    val_data = dict()
    train_data = dict()
    for k, trajs in all_data.items():
        num_val = int(cfg.dataset.val_split * len(trajs))
        print("Data level: {}, loaded {} episodes, using {} for validation, max/min traj length: {}/{}".format(
            k, len(trajs), num_val,
            max([len(t) for t in trajs]), min([len(t) for t in trajs])
            ))
        train_data[k] = trajs[:num_val]
        val_data[k] = trajs[num_val:]
    return ReplayDataset(cfg, train_data), ReplayDataset(cfg, val_data)

@hydra.main(config_name='config_rew', config_path='/home/mandi/ARM/irl')
def main(cfg: DictConfig) -> None: 
    
    train_dataset, val_dataset = load_dataset(cfg)  

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

    def compute_loss(batch):
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
        acc = torch.sum(low_logits < high_logits) / b
        return loss, acc, low_logits, high_logits

    
    if cfg.log_wb:
        run = wandb.init(project="ContextARM", name=cfg.run_name)
        cfg_dict = {
            k: v for k, v in cfg.items() if not 'dataset' in k
        }
        for key in ['dataset']:
            for sub_key in cfg[key].keys():
                cfg_dict[key+'/'+sub_key] = cfg[key][sub_key]
        run.config.update(cfg_dict)

    train_iter, val_iter = iter(train_dataset), iter(val_dataset)
    for step in range(cfg.itrs):
        batch = next(train_iter).to(device) # (num_levels=2, num_trajs, num_frames, 3, 128, 128
        loss, acc, low_logits, high_logits = compute_loss(batch)
        if torch.isnan(loss):
            print('Warning! got nan loss at step', step) 
            continue
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % cfg.log_freq == 0 or step == cfg.itrs - 1:
            log_step = step + 1 if step == cfg.itrs - 1 else step
            tolog = {'loss': loss.item(),
                'acc': acc.item(),
                'Train Step': log_step,
                'logits min': low_logits.min().item(), 
                'logits max': high_logits.max().item(),
                'logits mean': high_logits.mean().item(),
                }
            val_losses, val_accs, val_lows, val_hights = [], [], [], []
            for _ in range(cfg.dataset.val_steps):
                with torch.no_grad():
                    val_batch = next(val_iter).to(device)
                    val_loss, val_acc, val_low_logits, val_high_logits = compute_loss(val_batch)
                    val_losses.append(val_loss.item())
                    val_accs.append(val_acc.item())
                    val_lows.append(val_low_logits.min().item())
                    val_hights.append(val_high_logits.max().item())
            for key, val in zip(['val/loss', 'val/acc', 'val/logits min', 'val/logits max', 'val/logits mean'], \
                [val_losses, val_accs, val_lows, val_hights]):
                tolog[key] = np.mean(val)

            if cfg.log_wb:
                wandb.log(tolog)
            print(tolog)
            torch.save(mlp.state_dict(), f'{cfg.model_path}/{cfg.run_name}/{log_step}.pt')
 

if __name__ == "__main__":
    main()