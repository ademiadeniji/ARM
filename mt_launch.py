import os
import pickle
from os.path import join 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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
from yarr.utils.stat_accumulator import SimpleAccumulator

from arm import arm
from arm import c2farm
from arm.baselines import bc, td3, dac, sac
from arm.custom_rlbench_env import CustomRLBenchEnv, MultiTaskRLBenchEnv
import numpy as np

import hydra
import wandb 
import logging
from omegaconf import DictConfig, OmegaConf, ListConfig

from extar.runners.multi_env_runner import MultiTaskEnvRunner
from extar.runners.multi_task_trainer import MultiTaskPyTorchTrainer
from extar.utils.logger import MultiTaskAccumulator, WandbLogWriter
from extar.utils.rollouts import MultiTaskRolloutGenerator


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
    for tsk in cfg.rlbench.tasks:
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


def run_seed(cfg: DictConfig, env, cams, device, seed): # -> None:
    replay_ratio = None if cfg.framework.replay_ratio == 'None' else cfg.framework.replay_ratio
    replay_path = os.path.join(cfg.replay.path, cfg.short_names, cfg.method.name, 'seed%d' % seed)
    action_min_max = None

    if cfg.method.name == 'C2FARM': 
        
        replays = c2farm.launch_utils.create_and_fill_replays(
                cameras=cams, env=env, 
                save_dir=replay_path if cfg.replay.use_disk else None, **cfg.replay)
        agent = c2farm.launch_utils.create_agent(cfg, env) 
        
    else:
        raise NotImplementedError('Still need to support multi-task version of %s.' % cfg.method.name)
    
    stat_accum = MultiTaskAccumulator(
        cfg.rlbench.tasks, cfg.rlbench.eval_tasks, eval_video_fps=30, mean_only=True)
    logdir = join(cfg.log_path, 'seed%d' % seed)
    os.makedirs(logdir, exist_ok=False)
    
    weightsdir = join(logdir, 'weights')

    if action_min_max is not None:
        # Needed if we want to run the agent again
        os.makedirs(logdir, exist_ok=True)
        with open(join(logdir, 'action_min_max.pkl'), 'wb') as f:
            pickle.dump(action_min_max, f)
    OmegaConf.save( config=cfg, f=join(cfg.log_path, 'seed%d' % seed, 'config.yaml') )
    
    device_list = [ i for i in range(torch.cuda.device_count()) ]
    if len(device_list) > 1:
        print('Warning! Using multiple GPUs idxed: ', device_list)
        
    env_runner = MultiTaskEnvRunner( 
        env=env, 
        agent=agent, 
        replays=replays, 
        weightsdir=weightsdir,
        stat_accumulator=stat_accum, 
        rollout_generator=None,
        device_list=device_list,
        **cfg.env_runner )

    
    
    replays = {k: PyTorchReplayBuffer(r) for k, r in replays.items()}
    train_runner = MultiTaskPyTorchTrainer(
        agent=agent, 
        env_runner=env_runner, 
        replays=replays, 
        train_device=device, 
        device_list=device_list, 
        replay_buffer_sample_rates=cfg.framework.replay_sample_rates,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations, 
        logdir=logdir, 
        log_freq=cfg.framework.log_freq,  
        transitions_before_train=cfg.framework.transitions_before_train,
        weightsdir=weightsdir,
        save_freq=cfg.framework.save_freq,  
        replay_ratio=replay_ratio, 
        csv_logging=cfg.framework.csv_logging)

    if cfg.load:
            print('Warning! Loading back checkpoints from:', cfg.load_dir, cfg.load_step)
            train_runner.start(load_dir=join(cfg.load_dir, str(cfg.load_step)) )
             
    else:
        train_runner.start()
    del train_runner
    del env_runner
    torch.cuda.empty_cache()


@hydra.main(config_name='mt_confg', config_path='/home/mandi/ARM/conf')
def main(cfg: DictConfig): #-> None:
    torch.multiprocessing.set_start_method('spawn')
    cwd = os.getcwd()
    tasks_name = _gen_short_names(cfg)
    cfg.short_names = tasks_name
    log_path = join(cwd, tasks_name, cfg.method.name+cfg.run_name)
    os.makedirs(log_path, exist_ok=True)
    existing_seeds = len(list(filter(lambda x: 'seed' in x, os.listdir(log_path))))
    logging.info('Logging to:' + log_path)
    cfg.log_path = log_path 
    logging.info('\n' + OmegaConf.to_yaml(cfg))

    if cfg.framework.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:%d" % cfg.framework.gpu)
        torch.cuda.set_device(cfg.framework.gpu)
        torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
    else:
        print("Warning: Using CPU training \n")
        device = torch.device("cpu")
    logging.info('Using device %s.' % str(device))

    action_mode = ActionMode(
        ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME,
        GripperActionMode.OPEN_AMOUNT)
    
    from rlbench.backend import task
    from rlbench.backend.utils import task_file_to_task_class   

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    for task in cfg.rlbench.tasks:
        assert task in task_files, 'Task %s not recognised!.' % task

    cfg.rlbench.cameras = cfg.rlbench.cameras if isinstance(
        cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = _create_obs_config(cfg.rlbench.cameras, cfg.rlbench.camera_resolution)

    # train_envs = MultiTaskRLBenchEnv( cfg.rlbench.tasks, obs_config, action_mode, cfg.rlbench.single_env_cfg)
    # test_envs = MultiTaskRLBenchEnv( cfg.rlbench.test_tasks, obs_config, action_mode, cfg.rlbench.single_env_cfg)
    # contains both train and eval:
    env = MultiTaskRLBenchEnv(
        cfg.rlbench.tasks, 
        cfg.rlbench.eval_tasks, 
        obs_config, 
        action_mode, 
        **cfg.rlbench.single_env_cfg)
 
    run = wandb.init(project='rlbench', job_type='mt_launch')
    run.name = log_path
    cfg_dict = {}
    for key in ['rlbench', 'replay', 'framework', 'env_runner', 'trainer']:
 
        for sub_key in cfg[key].keys():
            cfg_dict[key+'/'+sub_key] = cfg[key][sub_key]
    run.config.update(cfg_dict)
    run.save()

    for seed in range(existing_seeds, existing_seeds + cfg.framework.seeds):
        logging.info('Starting seed %d.' % seed)
        run_seed(cfg, env, cfg.rlbench.cameras, device, seed)


if __name__ == '__main__':
    main()
