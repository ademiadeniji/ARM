"""A purely offline demonstration dataset used for context/representation learning"""

import numpy as np 
import random
import os
import json
import time 
import copy 
import torch
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
from natsort import natsorted
import logging
from functools import partial 
SHUFFLE_RNG = 2843014334
EXCLUDE_KEYS = [
    'last_img',
    'all_imgs'
    ] # use this solely for visualization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3,1,1))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3,1,1))
JITTER_FACTORS = {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1} 

def _load_and_maybe_resize(filename, size):
    image = Image.open(filename)
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image

def split_files(files, mode, splits):
    file_len = len(files)
    assert sum(splits) == 1 and all([0 <= s for s in splits]), "splits is not valid pdf!"

    order = [i for i in range(file_len)]
    random.Random(SHUFFLE_RNG).shuffle(order)
    pivot = int(len(order) * splits[0])
    if mode == 'train':
        order = order[:pivot]
    else:
        order = order[pivot:]
    return [files[o] for o in order]

def collate_by_id(collate_id, batch): 
    # this could either be collating by task or by variation
    """groups data by variations, so we can get per-task losses """
    batched_data = defaultdict(list)
    for b in batch:
        batched_data[b[collate_id]].append(
            {k:v for k, v in b.items() if (k != collate_id and k not in EXCLUDE_KEYS) }
        )
    for name, data in batched_data.items():
        batched_data[name] = default_collate(data)
    return batched_data

def visualize_batch(model_inp, inp_names=None, frame_idx=-1, filename='one_batch'):
    """ generate image for a batch of B,K,N,ch,H,W Tensor images """
    b, k, n, ch, img_h, img_w = model_inp.shape
    nrows = b
    ncols = k
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))
    for i in range(b):
        for j in range(k):
            img = model_inp[i,j,frame_idx] 
            img = img.cpu().numpy() * STD + MEAN
            assert len(img.shape) == 3, f'Got image shape: {img.shape}'
            row, col = i, j 
            axs[row, col].imshow(img.transpose(1,2,0))
            axs[row, col].axis('off')
            if inp_names:
                axs[row, col].set_title(inp_names[i][j])
             
        
        plt.tight_layout()
    plt.savefig(f'{filename}.png')


class RLBenchDemoDataset(Dataset):
    def __init__(
        self,
        image_size: List[int] = [128, 128],
        num_variations_per_task: int = -1, 
        num_episodes_per_variation: int = 10, 
        num_steps_per_episode: int = 2, 
        obs_config: ObservationConfig = ObservationConfig(),
        root_dir: str = '/home/mandi/all_rlbench_data',
        exclude_tasks: List[str] = [],
        include_tasks: List[str] = [],
        data_augs: dict = {}, 
        split: List[float] = [0.9, 0.1],
        mode: str = 'train',

        ):
        self._obs_config     = obs_config
        self._num_variations = num_variations_per_task
        self._num_episodes   = num_episodes_per_variation
        self._num_steps      = num_steps_per_episode

        all_names = sorted([ s.split('/')[-1] for s in  glob(join(root_dir, '*')) ])
        all_names = [n for n in all_names if '.txt' not in n] # only get task names
        
        assert not (len(exclude_tasks) > 0 and len(include_tasks) > 0), 'Must only specify task subset in exactly one way'
        if len(exclude_tasks) > 0:
            self._task_names = [name for name in all_names if name not in exclude_tasks]
        elif len(include_tasks) > 0:
            self._task_names = include_tasks #[name for name in all_names if name in include_tasks]
            for name in include_tasks:
                assert name in all_names, f"Task name: {name} not found locally"
        else:
            self._task_names = all_names 
        self._all_file_names = []

        prev = 0 # cursor 
        self._variation_idx_list = [] 
        self._task_idx_list = [] 
        self._task_variation_tree = []
        
        # create another map from each episode to variation/task it belongs to
        self._idx_to_task = []
        self._idx_to_variation = [] 
        self._idx_to_names = []

        for task_id, name in enumerate(self._task_names):
            variations = sorted( glob( join(root_dir, name, 'variation[0-9]*') ) )
            if self._num_variations > -1:
                variations = variations[: self._num_variations]
             
            all_variation_idxs = []
            listed_var_idxs = [] 
            if len(variations) == 0:
                print(f'Warning! No variation found for {name}')
            for var_id, one_var in enumerate(variations):
                episodes = sorted( glob( join(one_var, 'episodes/episode*') ) )
                if self._num_episodes > -1:
                    episodes = episodes[: self._num_episodes]
                
                assert len(episodes) == self._num_episodes, f'Not enough episodes found for {one_var}'
                episodes = split_files(episodes, mode, split)
                
                self._all_file_names.extend( episodes )
                num_new_episodes = len(episodes)
                new_indices = [i for i in range(prev, prev + num_new_episodes)]
                self._variation_idx_list.append( new_indices )
                prev += num_new_episodes
                all_variation_idxs.extend(new_indices)
                listed_var_idxs.append(new_indices)
                self._idx_to_variation.extend([len(self._variation_idx_list)-1 for _ in new_indices]) 
                # NOTE: these all come from one variation, but here count varation across all tasks
                self._idx_to_task.extend([task_id for _ in new_indices]) 
                self._idx_to_names.extend([ f'{name}_variation{var_id}_episode{eps_id}' for eps_id in range(num_new_episodes)])
            
            self._task_idx_list.append(all_variation_idxs)
            self._task_variation_tree.append(listed_var_idxs)
        
        self.total_count = len(self._all_file_names)
        
        logging.info(f'Loaded {len(self._task_idx_list)} distinct tasks, \
            after limited to {self._num_variations} variations for each task, \
            got {len(self._variation_idx_list)} total variations, \
            and {self.total_count} episodes.')
        # print(self._idx_to_names)

        jitters = {k: v * data_augs.get('strong_jitter', 0) for k,v in JITTER_FACTORS.items()}
        strong_jitter = ColorJitter(**jitters) # due to tasks design it's prolly not a good idea to jitter color
        self.grayscale = RandomGrayscale(data_augs.get("grayscale", 0))
        weak_scale = data_augs.get('weak_crop_scale', (0.7, 0.7))
        weak_ratio = data_augs.get('weak_crop_ratio', (0.8, 1))
        randcrop   = RandomResizedCrop(
            size=(128, 128), scale=weak_scale, ratio=weak_ratio)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.aug_transforms = transforms.Compose([ # normalize at the end
            ToTensor(),
            RandomApply([strong_jitter], p=0.05),
            self.grayscale,
            RandomHorizontalFlip(p=data_augs.get('flip', 0)),
            RandomApply(
                [GaussianBlur(kernel_size=5, sigma=data_augs.get('blur', (0.1, 2.0))) ], p=0.01),
            randcrop,
            self.normalize])
        self._mode = mode 
        self.val_transforms = transforms.Compose([ # normalize at the end
                            ToTensor(),
                            self.normalize])
         
    # see RLBench.rlbench.environment.Environment   

    def __len__(self): 
        return self.total_count 

    def __getitem__(self, idx):
        path = self._all_file_names[idx]
        obs = self._load_one_episode(path)
        return self._make_data(obs, idx)

    def _get_names(self, idx):
        return {
            'task_id': self._idx_to_task[idx], 
            'variation_id':  self._idx_to_variation[idx],
            'name':    self._idx_to_names[idx]
        }

    def _make_data(self, short_obs, idx):
        # short_obs should already only return a subset of obs in one episode, do data aug here
        data_dict = {}
        transforms = self.val_transforms if self._mode == 'val' else self.aug_transforms
        data_dict['front_rgb'] = torch.stack([
            transforms(
                np.array(ob.front_rgb)) for ob in short_obs])
        data_dict['last_img'] = short_obs[-1].front_rgb
        data_dict['all_imgs'] = [ob.front_rgb for ob in short_obs]
        data_dict.update(self._get_names(idx))
        # TODO: data_dict['front_depth'] = 
        return data_dict 

    def get_some_data(self, num_task=-1, num_vars=-1):
        """Return some tasks and Some of each's variations"""
        data_list = []
        task_idxs = self._task_variation_tree
        if num_task > 0:
            task_idxs = task_idxs[:num_task]    # limit num of tasks 
        for idx_list in task_idxs:
            if num_vars > 0:
                idx_list = idx_list[:num_vars]  # limit num of variations
            for var_idx in idx_list:
                data = self.__getitem__(var_idx[0]) 
                data_list.append(data)

        return data_list

    def sample_one_variation(self, task_id, variation_id):
        variations = self._task_variation_tree[task_id]
        episodes = variations[variation_id]
        eps_idx = np.random.randint(0, len(episodes))
        data = self.__getitem__( episodes[eps_idx] )
        return data 

    def sample_for_replay(self, task_ids, variation_ids):
        data_list = [
            self.sample_one_variation(task_id, variation_id) \
                for task_id, variation_id in zip(task_ids, variation_ids)
        ] 
        return data_list
    
    def get_idxs(self):
        return self._variation_idx_list, self._task_idx_list

    def _load_one_task(self, path):
        variation_paths = sorted( glob( join(path, 'variation*') ) )
        assert len(variation_paths) >= self._num_variations, f'Got only {len(variation_paths)} saved locally for this task {path}'
        one_task_demos  = []
        for vpath in variation_paths[:self._num_variations]:
            one_task_demos.append( self._load_one_variation(vpath) ) 

        return one_task_demos

    def _load_one_variation(self, path):
        episode_paths = sorted( glob( join(path, 'episodes/*') )  )
        assert len(episode_paths) >= self._num_episodes, f'Got only {len(episode_paths)} saved locally for this task variation {path}'
        demos = [self._load_one_episode(path) for path in episode_paths[:self._num_episodes]]
        return demos 
    
    def _load_one_episode(self, path):
        """see  rlbench.utils.get_stored_demos """
        with open(join(path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pkl.load(f)
        
        # TODO: add more cameras, only front for now 
        front_size = self._obs_config.front_camera.image_size

        aval_steps = len(obs) 
            
        if self._num_steps == 2:
            take_steps = [0, aval_steps-1]
        elif self._num_steps == 1:
            take_steps = [aval_steps-1]
        else:
            take_steps = [0] + \
                list(np.random.choice(range(1, aval_steps-1), self._num_steps) ) + \
                [aval_steps-1]

        for name in [FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER]: #, FRONT_MASK_FOLDER]:
            folder = join(path, name) 
            assert aval_steps ==  len(listdir(folder)), \
                f'Broken dataset assumption on folder {folder}, \
                should contain {aval_steps} steps but got {len(listdir(folder))}'

        
        short_obs = []
        for i in take_steps:
            si = IMAGE_FORMAT % i 
            obs[i].front_rgb = _load_and_maybe_resize(
                join(path, FRONT_RGB_FOLDER, si), front_size)   
            obs[i].front_depth = _load_and_maybe_resize(
                join(path, FRONT_DEPTH_FOLDER, si), front_size) 
            # obs[i].front_mask = _load_and_maybe_resize(
            #     join(path, FRONT_MASK_FOLDER, si), front_size)  

            # Remove low dim info if necessary
            if not self._obs_config.joint_velocities:
                obs[i].joint_velocities = None
            if not self._obs_config.joint_positions:
                obs[i].joint_positions = None
            if not self._obs_config.joint_forces:
                obs[i].joint_forces = None
            if not self._obs_config.gripper_open:
                obs[i].gripper_open = None
            if not self._obs_config.gripper_pose:
                obs[i].gripper_pose = None
            if not self._obs_config.gripper_joint_positions:
                obs[i].gripper_joint_positions = None
            if not self._obs_config.gripper_touch_forces:
                obs[i].gripper_touch_forces = None
            if not self._obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None
            short_obs.append(obs[i])
        
        return short_obs 

    def sanity_check(self):
        imgs = []
        task_names = []
        for idxs in self._variation_idx_list:
            grab_one = self._load_one_episode(
                self._all_file_names[idxs[0]])
            obs = grab_one[-1]
            imgs.append(obs.front_rgb)
            splitted = self._all_file_names[idxs[0]].split('/')[-4:]
            splitted.remove('episodes')
            #print(splitted) 
            name = splitted[0] #'_'.join(splitted) 
            task_names.append( name )
        print(f'Grabbed a total of {len(imgs)} images from all variations')
        nrows = 10
        ncols = int(len(imgs) / nrows)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))
        for i, img in enumerate(imgs):
            row, col = i // nrows, i % ncols
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
            axs[row, col].set_title(task_names[i], fontsize=15)
        
        plt.tight_layout()
        plt.savefig('all_tasks.png')

    def sample_context_batch(self, batch_dim, num_variations, sample_mode='variation'):
        """ samples b, k, num_steps, 3, 128, 128 """


class MultiTaskDemoSampler(Sampler):
    """
    This is a custom BatchSampler, returns batch of indices grouped by task variations, 
    each task variation uses a shuffled list of idxs to yield fixed size-N number of indices 
    """
    def __init__(
        self,
        variation_idxs_list: List[int] = [], # 1-1 maps from each variation to idx in dataset e.g. [[0,1], [2,3], [4,5]] belongs to 3 variations but 2 tasks
        task_idxs_list: List[int] = [],      # 1-1 maps from each task to all its variations e.g. [[0,1,2,3], [4,5]] collects variations by task
        batch_dim: int = 2,       # dimension B, depends on sample mode: could either be num of distinct tasks in a batch OR num of variations 
        samples_per_variation: int = 2, # dimension 'N' 
        drop_last: bool = False, 
        sample_mode: str = 'variation', # either sample by task or by variations 
        ):
        """
        Note that;
            - In local dataset, all variations should default to have the same number of demo episodes 
        """
        self._batch_dim   = batch_dim # can either be num of vars or tasks
        self._samples_per_variation = samples_per_variation 
        self._sample_mode = sample_mode 

        if sample_mode == 'task':
            tocopy = task_idxs_list 
        elif sample_mode == 'variation':
            tocopy = variation_idxs_list 
            # NOTE: it's possible that different variations in the same task get sampled together
        else:
            raise ValueError(f'Got unsupported request to sample a batch by {sample_mode}')
         
        self._batch_idx_list = copy.deepcopy(tocopy) 
        [random.shuffle(idxs) for idxs in self._batch_idx_list]
          
        self._num_total_variations = len(variation_idxs_list)
        self._num_total_tasks = len(task_idxs_list) 
        self._list_idxs = [i for i in range(len(self._batch_idx_list))]
        for i, idxs in enumerate(self._batch_idx_list):
            assert len(idxs) >= self._samples_per_variation, f'Variation {i} does not have enough samples for even one iteration!'
        
        self.drop_last = drop_last 
        # logging.info(f'Sampling from a dataset of \
        #     {len(task_idxs_list)} distinct tasks, \
        #     {self._num_total_variations} total variations, \
        #     using {len(self._list_idxs)} iterators') 

    def __iter__(self):  
        b_dim = self._batch_dim
        while len(self._list_idxs) >= b_dim:  
            batch = []
            var_idxs = np.random.choice(
                self._list_idxs, 
                size=b_dim, 
                replace=False)
            for idx in var_idxs: 
                episode_idxs = self._batch_idx_list[idx][:self._samples_per_variation]
                batch.append(episode_idxs)
                self._batch_idx_list[idx] = self._batch_idx_list[idx][self._samples_per_variation:]
                if len(self._batch_idx_list[idx]) < self._samples_per_variation:
                    self._list_idxs.remove(idx)
                    #print(f'removing {idx}')
            assert len(batch) == b_dim 
            yield sum(batch, [])
            batch = []      
        if not self.drop_last:
            batch = [ self._batch_idx_list[idx][:self._samples_per_variation] for idx in self._list_idxs]
            if len(batch) > 0:
                yield sum(batch, [])
                batch = []
  

class PyTorchIterableDemoDataset(IterableDataset):
    """ See YARR.replay.wrappers, work around the need for Sampler so multiprocessing can pickle"""
    def __init__(
        self, 
        batch_dim: int = 2,       # dimension B, depends on sample mode: could either be num of distinct tasks in a batch OR num of variations 
        samples_per_variation: int = 2, # dimension 'N'  
        sample_mode: str = 'variation', # either sample by task or by variations 
        demo_dataset: RLBenchDemoDataset = None ,
        ):
        
        self._demo_dataset = demo_dataset 
        self._batch_dim   = batch_dim # can either be num of vars or tasks
        self._samples_per_variation = samples_per_variation 
        self._sample_mode = sample_mode 

        if sample_mode == 'task':
            tocopy = demo_dataset._task_idx_list 
        elif sample_mode == 'variation':
            tocopy = demo_dataset._variation_idx_list 
            # NOTE: it's possible that different variations in the same task get sampled together
        else:
            raise ValueError(f'Got unsupported request to sample a batch by {sample_mode}')
         
        self._sample_mode = sample_mode 
        self._collate_id = sample_mode + '_id'
        self._batch_idx_list = copy.deepcopy(tocopy) 
        [random.shuffle(idxs) for idxs in self._batch_idx_list]
          
        self._num_total_variations = len(demo_dataset._variation_idx_list)
        self._num_total_tasks = len(demo_dataset._task_idx_list) 
        self._list_idxs = [i for i in range(len(self._batch_idx_list))]
        for i, idxs in enumerate(self._batch_idx_list):
            assert len(idxs) >= self._samples_per_variation, f'Variation {i} does not have enough samples for even one iteration!'
    
    def _generator(self):
        while True:
            yield self.sample_context_batch()
    
    def sample_context_batch(self):
        # taken from **idx** sampler code, but no exhaustion or removing
        batch = []
        var_idxs = np.random.choice(
            self._list_idxs, 
            size=self._batch_dim, 
            replace=False)
        for idx in var_idxs: 
            episode_idxs = np.random.choice(self._batch_idx_list[idx], size=self._samples_per_variation, replace=False) 
            batch.append(episode_idxs)
        assert len(batch) == self._batch_dim

        data_batch = []
        for episode_idxs in batch:
            # keys = self._demo_dataset.__getitem__(episode_idxs[0]).keys()
            # collated = {
            #     key: torch.cat
            # }
            # for idx in episode_idxs: # this is absolute indices
            #     data_batch.append(
            #         self._demo_dataset.__getitem__(idx)
            #     )
            all_eps_data = [self._demo_dataset.__getitem__(idx) for idx in episode_idxs]
            data_batch.extend(all_eps_data) 
        return collate_by_id(self._collate_id, data_batch)

    def sample_for_replay(self, task_ids, variation_ids):
        return self._demo_dataset.sample_for_replay(task_ids, variation_ids)

    def __iter__(self):
        return iter(self._generator())



if __name__ == '__main__':
    one_cam = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=[128,128],
        render_mode=RenderMode.OPENGL)

    tester_config = ObservationConfig(
        front_camera=one_cam)
    demo_dataset = RLBenchDemoDataset(
        num_variations_per_task=-1, 
        num_episodes_per_variation=20, 
        num_steps_per_episode=1, 
        obs_config=tester_config,
        root_dir='/home/mandi/all_rlbench_data',
        exclude_tasks=[],
        data_augs={},
        mode='train'
        )

    # one_load = demo_dataset[0]
    #print( [(k, v.shape) for k, v in one_load.items()] )
    #demo_dataset.sanity_check()
    # for path in demo_dataset._all_file_names:
    #     demo_dataset._load_one_episode(path)

    # variation_idxs, task_idxs = demo_dataset.get_idxs()
    # print(f'total variation idxs {len(variation_idxs)}, total task ids: {len(task_idxs)}')
    # sample_mode = 'variation'
    # sampler = MultiTaskDemoSampler(
    #     variation_idxs_list=variation_idxs, # 1-1 maps from each variation to idx in dataset e.g. [[0,1], [2,3], [4,5]] belongs to 3 variations but 2 tasks
    #     task_idxs_list=task_idxs,      # 1-1 maps from each task to all its variations e.g. [[0,1,2,3], [4,5]] collects variations by task
    #     batch_dim=5, # dimension 'B'
    #     samples_per_variation=6, # dimension 'N' 
    #     drop_last=True,
    #     sample_mode=sample_mode,
    # )
    # collate_func = partial(collate_by_id, sample_mode+'_id')
    # train_loader = DataLoader(
    #         demo_dataset, 
    #         batch_sampler=sampler,
    #         num_workers=5,
    #         worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w),
    #         collate_fn=collate_func,  
             
    #         )
    iterable_dataset = PyTorchIterableDemoDataset(
        batch_dim=2,       # dimension B, depends on sample mode: could either be num of distinct tasks in a batch OR num of variations 
        samples_per_variation=1, # dimension 'N'  
        sample_mode='variation', # either sample by task or by variations 
        demo_dataset=demo_dataset,
    )
     
        
    
    # t = time.time()
    # count = 0
    # epochs = 3
     
    # for e in range(epochs):
    #     for s in train_loader:
    #         count += 1 
    #     print('epoch counting', e, count)
    # spent = time.time() - t
    # print('finished loading one epoch, count:', count, spent)
    
    # for data in train_loader:
    #     model_inp = torch.stack([v['front_rgb'] for k, v in data.items()])
    #     inp_names = [v['name'] for k, v in data.items()] 
 
    #     visualize_batch(model_inp, inp_names=inp_names)
    #     break
     
         
    