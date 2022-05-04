import torch 
import numpy as np 
import random
import os
import json
import time 
import copy 
from os import listdir
from os.path import join, expanduser 
from torch.utils.data import Sampler, SubsetRandomSampler, RandomSampler, BatchSampler
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.data._utils.collate import default_collate 
from torch.multiprocessing import cpu_count
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms import functional as TvF
from torchvision.transforms.functional import resized_crop

import pickle as pkl
from collections import defaultdict, OrderedDict 
from glob import glob
import matplotlib.pyplot as plt

from rlbench import ObservationConfig, CameraConfig
from pyrep.const import RenderMode
from rlbench.demo import Demo
from rlbench.backend.const import *
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from typing import List
from PIL import Image
from natsort import natsorted # !!!
import logging

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
try:
    from torchvision.transforms import InterpolationMode 
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ReplayDataset(IterableDataset):
    """ New(0423): new multi-task version """
    def __init__(self, cfg, all_data: defaultdict) -> None:
        super(ReplayDataset).__init__()
        self.all_data = all_data
        self.levels = list(all_data.keys())
        self.level_idxs = [i for i in range(len(self.levels))]
        self.batch_size = cfg.dataset.bsize
        self.num_trajs = cfg.dataset.ntrajs 
        self.num_frames_per_traj = cfg.dataset.nframes
        self.sample_expert_prob = cfg.dataset.sample_expert_prob
        if len(self.levels) == 1 and self.levels[0] == 'expert':
            self.sample_expert_prob = 1
        self.sample_success_prob = cfg.dataset.sample_success_prob
        if 'success' not in self.levels:
            self.sample_success_prob = 0 
        self.aug_process = Compose([
            Resize(int(cfg.dataset.resize), interpolation=BICUBIC),
            RandomResizedCrop(224, scale=(0.7, 1.0)),
            RandomHorizontalFlip(cfg.dataset.flip_prob),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.cfg = cfg.dataset
        self.concat_batch = cfg.dataset.concat_batch
        if self.concat_batch:
            self.num_trajs = int(self.num_trajs / 2)
            print('Concatenating batch, half the num of trajs into {self.num_trajs}')
            

    def _generator(self):
        while True:
            yield self.sample_batch()

    def sample_concat_batch(self):
        """ Sample both the temporal and fail/success trajs """
        temporal_batch = self.sample_success_batch() # Either expert OR success!
        binary_batch = self.sample_binary_batch()
        return torch.cat([temporal_batch, binary_batch], dim=1)

    def sample_batch(self):
        """Return batch (num_levels=2, num_trajs, num_frames, 3, 128, 128)"""
        if self.concat_batch:
            return self.sample_concat_batch()
        if random.random() < self.sample_expert_prob:
            return self.sample_success_batch()
        else:
            return self.sample_binary_batch()
        
    def sample_binary_batch(self):
        batch = []
        if self.cfg.fail_only:
            idxs = [0] + np.random.choice([i for i in self.level_idxs if i != 0], size=1).tolist()
        else:
            random.shuffle(self.level_idxs)
            idxs = sorted(self.level_idxs[:2]) 
        for level_idx in idxs:
            level_data = []
            data = self.all_data[self.levels[level_idx]]
            traj_idxs = np.random.choice(len(data), size=self.num_trajs, replace=(len(data) < self.num_trajs))

            for traj_idx in traj_idxs:
                traj = data[traj_idx]
                frame_idxs = sorted(np.random.choice(len(traj), size=self.num_frames_per_traj, replace=(len(traj) < self.num_frames_per_traj)))
                frames = [self.aug_process(
                    Image.fromarray(
                        np.uint8(traj[i]) 
                            )) for i in frame_idxs]
                # frames = [self.aug_process(traj[i]) for i in frame_idxs]
                level_data.append(torch.stack(frames))
            level_data = torch.stack(level_data, dim=0)
            batch.append(level_data)

        return torch.stack(batch)

    def sample_success_batch(self):
        """ split the expert level into earlier v.s. later in timestep """
        level = 'success' if random.random() < self.sample_success_prob else 'expert'
        data = self.all_data[level]
        data = [d for d in data if len(d) >= self.num_frames_per_traj * 2]
        traj_idxs = np.random.choice(len(data), size=self.num_trajs, replace=(len(data) < self.num_trajs))
        low_batch, high_batch = [], [] 
        for traj_idx in traj_idxs:
            traj = data[traj_idx]
            assert len(traj) >= self.num_frames_per_traj * 2
            low_idx = np.random.choice(
                int(len(traj)/2) , 
                size=self.num_frames_per_traj, 
                replace=(int(len(traj)/2) < self.num_frames_per_traj)
                )
            low_idx = sorted(low_idx)
            assert low_idx[-1] == max(low_idx)
            high_idx = np.random.choice(
                range(low_idx[-1] + 1, len(traj)), 
                size=self.num_frames_per_traj, 
                replace=(len(traj) - low_idx[-1] - 1 < self.num_frames_per_traj)
                )
            high_idx = sorted(high_idx)

            for frame_idxs, batch in zip([low_idx, high_idx], [low_batch, high_batch]):
                if level == 'success':
                    frames = [self.aug_process(
                        Image.fromarray( np.uint8(traj[i]) )) for i in frame_idxs] 
                else:
                    frames = [self.aug_process(traj[i]) for i in frame_idxs] 
                batch.append(torch.stack(frames))
        low_batch = torch.stack(low_batch, dim=0)
        high_batch = torch.stack(high_batch, dim=0)
        return torch.stack([low_batch, high_batch])

    def __iter__(self):
        return iter(self._generator())

