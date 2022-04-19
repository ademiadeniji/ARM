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

        self.aug_process = Compose([
            Resize(int(cfg.dataset.resize), interpolation=BICUBIC),
            RandomResizedCrop(224, scale=(0.7, 1.0)),
            RandomHorizontalFlip(cfg.dataset.flip_prob),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _generator(self):
        while True:
            yield self.sample_batch()

    def sample_batch(self):
        """Return batch (num_levels=2, num_trajs, num_frames, 3, 128, 128)"""
        
        if random.random() < self.sample_expert_prob:
            return self.sample_expert_batch()
        
        batch = []
        random.shuffle(self.level_idxs)
        idxs = sorted(self.level_idxs[:2]) 
        for level_idx in idxs:
            level_data = []
            data = self.all_data[self.levels[level_idx]]
            traj_idxs = np.random.choice(len(data), size=self.num_trajs, replace=False)

            for traj_idx in traj_idxs:
                traj = data[traj_idx]
                frame_idxs = sorted(np.random.choice(len(traj), size=self.num_frames_per_traj, replace=False))
                frames = [self.aug_process(
                    Image.fromarray(
                        np.uint8(traj[i]) 
                            )) for i in frame_idxs]
                # frames = [self.aug_process(traj[i]) for i in frame_idxs]
                level_data.append(torch.stack(frames))
            level_data = torch.stack(level_data, dim=0)
            batch.append(level_data)

        return torch.stack(batch)

    def sample_expert_batch(self):
        """ split the expert level into earlier v.s. later in timestep """
        data = self.all_data['expert']
        traj_idxs = np.random.choice(len(data), size=self.num_trajs, replace=False)
        low_batch, high_batch = [], [] 
        for traj_idx in traj_idxs:
            traj = data[traj_idx]
            assert len(traj) > self.num_frames_per_traj * 2
            low_idx = np.random.choice(len(traj) - self.num_frames_per_traj * 2, size=self.num_frames_per_traj, replace=False)
            low_idx = sorted(low_idx)
            assert low_idx[-1] == max(low_idx)
            high_idx = np.random.choice(
                range(low_idx[-1] + 1, len(traj)), size=self.num_frames_per_traj, replace=False)
            high_idx = sorted(high_idx)

            for frame_idxs, batch in zip([low_idx, high_idx], [low_batch, high_batch]):
                frames = [self.aug_process(traj[i]) for i in frame_idxs] 
                batch.append(torch.stack(frames))
        low_batch = torch.stack(low_batch, dim=0)
        high_batch = torch.stack(high_batch, dim=0)
        return torch.stack([low_batch, high_batch])



    def __iter__(self):
        return iter(self._generator())