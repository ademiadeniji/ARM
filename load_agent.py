""" Load back a q-attention agent for video/TSNE/rollout reward collection"""
import os 
from os.path import join, exists
import hydra 
import numpy as np 
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from pyrep.const import RenderMode
from rlbench import ObservationConfig, CameraConfig
from multiprocessing import cpu_count
from arm.demo_dataset import MultiTaskDemoSampler, RLBenchDemoDataset, collate_by_id
from arm.models.slowfast  import TempResNet
from arm.models.utils import make_optimizer
from einops import rearrange, reduce, repeat, parse_shape
import logging
from functools import partial 
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import wandb
from glob import glob 
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10, cividis, turbo 
import pandas as pd
from io import BytesIO
import base64
import PIL
from PIL import Image
from natsort import natsorted

from arm import arm
from arm import c2farm
from arm.baselines import bc, td3, dac, sac
from arm.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import numpy as np

# def collect_one_agent_one_task(agent, env, )

def collect_rollout(agent, weightsdir, cfg):
    env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes, task_names=tasks, observation_config=obs_config,
        action_mode=ACTION_MODE, dataset_root=cfg.rlbench.demo_path,
        episode_length=cfg.rlbench.episode_length, headless=True)



@hydra.main(config_name='config_loadback', config_path='conf')
def main(cfg: DictConfig) -> None: 
    load_dir = join(cfg.logdir, cfg.load_dir)
    if 'seed' not in load_dir:
        print(f'Appending seed {cfg.seed} to {load_dir}')
        load_dir = join(load_dir, f'seed{cfg.seed}')
    assert exists(load_dir). f'agent run {load_dir} does not exist!'

    agent_cfg = OmegaConf.load( join(load_dir, 'config.yaml') )
    if agent_cfg.mt_only:
        agent = c2farm.launch_utils.create_agent(agent_cfg, env)
    else:
        agent = c2farm.launch_utils.create_agent_with_context(agent_cfg, env)

    weightsdir = natsorted( glob(load_dir, 'weights/*') )
    
    weightsdir = weightsdir[cfg.load_step:]
    print(f'Load last {cfg.load_step} agent, saved at steps:', weightsdir)

    if cfg.evaluate:
        logging.info(f'Evaluating {len(weightsdir)} agents on tasks: ', cfg.eval_tasks)
