"""
Mandi(0803): use this to launch meta-rl agents 
"""
import os
import pickle

from arm.custom_rlbench_env_multitask import CustomMultiTaskRLBenchEnv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["QT_LOGGING_RULES" ] = '*.debug=false;qt.qpa.*=false'

from typing import List
import torch
from pyrep.const import RenderMode
from rlbench import CameraConfig, ObservationConfig, ArmActionMode
from rlbench.action_modes import ActionMode, GripperActionMode
from rlbench.backend import task
from rlbench.backend.utils import task_file_to_task_class
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.pytorch_train_runner import PyTorchTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator, MultiTaskAccumulator

from arm import arm
from arm import c2farm
from arm.baselines import bc, td3, dac, sac
from arm.custom_rlbench_env import CustomRLBenchEnv
import numpy as np

import hydra
import logging
from omegaconf import DictConfig, OmegaConf, ListConfig
from os.path import join 
import wandb 

from yarr.utils.multitask_rollout_generator import RolloutGenerator
from launch_multitask import _gen_short_names, _create_obs_config, _modify_action_min_max
 
def run_seed(cfg: DictConfig, env, cams, device, seed, tasks) -> None:
    train_envs = cfg.framework.train_envs
    replay_ratio = None if cfg.framework.replay_ratio == 'None' else cfg.framework.replay_ratio
    replay_split = [1]
    replay_path = os.path.join(cfg.replay.path, cfg.rlbench.task, cfg.method.name, 'seed%d' % seed)
    action_min_max = None

    if cfg.method.name == 'C2FARM':
        if cfg.replay.share_across_tasks: 
            logging.info(f'Using only one replay for multiple tasks, one batch size: {cfg.replay.batch_size}')
            r = c2farm.launch_utils.create_replay(
                cfg.replay.batch_size, 
                cfg.replay.timesteps,
                cfg.replay.prioritisation,
                replay_path if cfg.replay.use_disk else None, cams, env,
                cfg.method.voxel_sizes)
            for task in tasks:
                c2farm.launch_utils.fill_replay(
                r, task, env, cfg.rlbench.demos,
                cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
                cams, cfg.rlbench.scene_bounds,
                cfg.method.voxel_sizes, cfg.method.bounds_offset,
                cfg.method.rotation_resolution, cfg.method.crop_augmentation)
            replays = [r]
        else:
            replays = []
            for task in tasks:
                r = c2farm.launch_utils.create_replay(
                    cfg.replay.batch_size, cfg.replay.timesteps,
                    cfg.replay.prioritisation,
                    replay_path if cfg.replay.use_disk else None, cams, env,
                    cfg.method.voxel_sizes)
                replays.append(r)
                c2farm.launch_utils.fill_replay(
                    r, task, env, cfg.rlbench.demos,
                    cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
                    cams, cfg.rlbench.scene_bounds,
                    cfg.method.voxel_sizes, cfg.method.bounds_offset,
                    cfg.method.rotation_resolution, cfg.method.crop_augmentation)

        agent = c2farm.launch_utils.create_agent_with_context(cfg, env)
    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    wrapped_replays = [PyTorchReplayBuffer(r) for r in replays]
    stat_accum = MultiTaskAccumulator(cfg.short_names, eval_video_fps=30)

    #cwd = os.getcwd()
    #weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    #logdir = os.path.join(cwd, 'seed%d' % seed)
    logdir = join(cfg.log_path, 'seed%d' % seed)
    os.makedirs(logdir, exist_ok=False)
    OmegaConf.save( config=cfg, f=join(logdir, 'config.yaml') )
    
    weightsdir = join(logdir, 'weights')
    

    if action_min_max is not None:
        # Needed if we want to run the agent again
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, 'action_min_max.pkl'), 'wb') as f:
            pickle.dump(action_min_max, f)

    device_list = [ i for i in range(torch.cuda.device_count()) ]
    assert len(device_list) > 1, 'Must use multiple GPUs'
    env_runner = EnvRunner(
        train_env=env, agent=agent, train_replay_buffer=replays,
        num_train_envs=train_envs,
        num_eval_envs=cfg.framework.eval_envs,
        episodes=99999,
        episode_length=cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        device_list=device_list[2:]) # ugly hack, leave at least 2 gpus for the agents

    # run = wandb.init(project='rlbench', job_type='patched', sync_tensorboard=True)

    train_runner = PyTorchTrainRunner(
        agent, env_runner,
        wrapped_replays, device, replay_split, stat_accum,
        iterations=cfg.framework.training_iterations,
        save_freq=cfg.framework.save_freq, 
        log_freq=cfg.framework.log_freq, 
        logdir=logdir,
        weightsdir=weightsdir,
        replay_ratio=replay_ratio,
        transitions_before_train=cfg.framework.transitions_before_train,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging)
    train_runner.start()
    del train_runner
    del env_runner
    torch.cuda.empty_cache()


@hydra.main(config_name='config_multitask', config_path='conf')
def main(cfg: DictConfig) -> None:
    
    logging.info('\n' + OmegaConf.to_yaml(cfg))
    tasks_name = _gen_short_names(cfg)
    
    if cfg.framework.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:%d" % cfg.framework.gpu)
        torch.cuda.set_device(cfg.framework.gpu)
        torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    logging.info('Using device %s.' % str(device))

    action_mode = ActionMode(
        ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME,
        GripperActionMode.OPEN_AMOUNT)

    tasks = cfg.tasks 
    task_names = [SHORT_NAMES.get(tsk) for tsk in tasks]
    cfg.short_names = task_names 
    task_classes = [task_file_to_task_class(t) for t in tasks]

    cfg.rlbench.cameras = cfg.rlbench.cameras if isinstance(
        cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = _create_obs_config(cfg.rlbench.cameras,
                                    cfg.rlbench.camera_resolution)

    env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes, task_names=task_names, observation_config=obs_config,
        action_mode=action_mode, dataset_root=cfg.rlbench.demo_path,
        episode_length=cfg.rlbench.episode_length, headless=True)

    cwd = os.getcwd()
    log_path = join(cwd, tasks_name, cfg.method.name+'-'+cfg.run_name)
    os.makedirs(log_path, exist_ok=True)
    #logging.info('CWD:' + os.getcwd())
    existing_seeds = len(list(filter(lambda x: 'seed' in x, os.listdir(log_path))))
    logging.info('Logging to:' + log_path)
    cfg.log_path = log_path 
    logging.info('\n' + OmegaConf.to_yaml(cfg))

    for seed in range(existing_seeds, existing_seeds + cfg.framework.seeds):
        logging.info('Starting seed %d.' % seed)
        run_seed(cfg, env, cfg.rlbench.cameras, device, seed, tasks)


if __name__ == '__main__':
    main()

