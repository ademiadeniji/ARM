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

SHORT_NAMES = {
    'pick_up_cup':          'cup',
    'phone_on_base':        'phone',
    'pick_and_lift':        'lift',
    'put_rubbish_in_bin':   'rubbish',
    'reach_target':         'target',
    'stack_wine':           'wine', 
    'take_lid_off_saucepan': 'sauce',
    'take_umbrella_out_of_umbrella_stand': 'umbrella',
    'meat_off_grill':       'grill',
    'put_groceries_in_cupboard': 'grocery',
    'take_money_out_safe':  'safe',
    'unplug_charger':       'charger'
}

def _gen_short_names(cfg: DictConfig): # just for logging dirs
    names = []
    cfg.tasks = sorted(cfg.tasks)
    for tsk in cfg.tasks:
        names.append(SHORT_NAMES[tsk])
    names = sorted(names)
    return f"{len(names)}tasks-" + "-".join(names)

def _create_obs_config(camera_names: List[str], camera_resolution: List[int]):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=False,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


def _modify_action_min_max(action_min_max):
    # Make translation bounds a little bigger
    action_min_max[0][0:3] -= np.fabs(action_min_max[0][0:3]) * 0.2
    action_min_max[1][0:3] += np.fabs(action_min_max[1][0:3]) * 0.2
    action_min_max[0][-1] = 0
    action_min_max[1][-1] = 1
    action_min_max[0][3:7] = np.array([-1, -1, -1, 0])
    action_min_max[1][3:7] = np.array([1, 1, 1, 1])
    return action_min_max


def run_seed(cfg: DictConfig, env, cams, device, seed, tasks) -> None:
    train_envs = cfg.framework.train_envs
    replay_ratio = None if cfg.framework.replay_ratio == 'None' else cfg.framework.replay_ratio
    replay_split = [1]
    replay_path = os.path.join(cfg.replay.path, cfg.rlbench.task, cfg.method.name, 'seed%d' % seed)
    action_min_max = None

    if cfg.method.name == 'C2FARM':
        if cfg.replay.share_across_tasks:
            
            total_size = int(len(tasks) * cfg.replay.batch_size)
            logging.info(f'New: try using one replay for multiple tasks, one batch size is aggregated to {total_size}')
            cfg.replay.batch_size = total_size
            r = c2farm.launch_utils.create_replay(
                total_size, cfg.replay.timesteps,
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

        agent = c2farm.launch_utils.create_agent(cfg, env)
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
    if len(device_list) > 1:
        print('Warning! Using multiple GPUs idxed: ', device_list)
    env_runner = EnvRunner(
        train_env=env, agent=agent, train_replay_buffer=replays,
        num_train_envs=train_envs,
        num_eval_envs=cfg.framework.eval_envs,
        episodes=99999,
        episode_length=cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        device_list=device_list,
        share_buffer_across_tasks=cfg.replay.share_across_tasks)

    resume_dir = None
    if cfg.resume:
        resume_dir = join(cfg.resume_path, cfg.resume_run, 'weights', str(cfg.resume_step))
        assert os.path.exists(resume_dir), 'Cannot find the weights saved at path: '+resume_dir
        cfg.framework.resume_dir = resume_dir 
    
    if cfg.framework.wandb_logging:
        run = wandb.init(project='MTARM', job_type='launch')
        run.name = cfg.log_path
        cfg_dict = {}
        for key in ['rlbench', 'replay', 'framework', 'contexts']:
            for sub_key in cfg[key].keys():
                cfg_dict[key+'/'+sub_key] = cfg[key][sub_key]
        run.config.update(cfg_dict)
        run.save()


    # trainer doesn't know if this is meta-rl agent 
    train_runner = PyTorchTrainRunner(
        agent, env_runner,
        wrapped_replays, device, replay_split, stat_accum,
        iterations=cfg.framework.training_iterations,
        save_freq=cfg.framework.log_freq, 
        log_freq=cfg.framework.log_freq, 
        logdir=logdir,
        weightsdir=weightsdir,
        replay_ratio=replay_ratio,
        transitions_before_train=cfg.framework.transitions_before_train,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        wandb_logging=cfg.framework.wandb_logging
        )
    
    
    train_runner.start(resume_dir)
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

