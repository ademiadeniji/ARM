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
from glob import glob 
DATA_DIR = '/home/mandi/all_rlbench_data' # /shared/mandi/all_rlbench_data 
DEMOS_PER_VARIATIONS = 10 

SKIP_TASKS = {
    'block_pyramid':            'pyramid',
    'weighing_scales':          'scales',
    'pick_up_cup':              'cup',
    'phone_on_base':            'phone',
    'pick_and_lift':            'lift',
    'put_rubbish_in_bin':       'rubbish',
    'reach_target':             'target',
    'stack_wine':               'wine', 
    'take_lid_off_saucepan':    'sauce',
    'take_umbrella_out_of_umbrella_stand': 'umbrella',
    'meat_off_grill':           'grill',
    'put_groceries_in_cupboard': 'grocery',
    'take_money_out_safe':      'safe',
    'unplug_charger':           'charger', ## testing water to see if everything below are solvable via single camera 
    'open_door':                'door',  # yes 
    'take_off_weighing_scales': 'scales', # no, 10% errors out after 2000 steps: rlbench.task_environment.TaskEnvironmentError: Could not place the task take_off_weighing_scales in the scene. This should not happen, please raise an issues on this task.
    'hit_ball_with_queue':      'queue', # no, 10%
    'press_switch':             'switch', # 100before got ~50%, 2cameras got 80%
    'open_box':                 'box',   # no, ~10%
    'light_bulb_out':           'bulb',   # no, floating bulb in the air?
    'light_bulb_in':            'bulb_in',
    'put_knife_on_chopping_board': 'knife', # yes
    'straighten_rope':          'rope', # NO 
    'hockey':                   'hockey',
    'open_drawer':              'drawer', # no, ~0% (wrong camera?)
    'beat_the_buzz':            'buzz',
    'open_fridge':              'fridge', # no, ~0%
    'take_usb_out_of_computer': 'usb',    # yes, >90%
    'push_buttons':             'buttons', # yes! >~100%
    'change_channel':           'channel',  # no, ~0%
    'reach_and_drag':           'drag',     # no, ~0%
    'close_laptop_lid':        'laptop',    # yes, 70%
    'open_oven':                'oven',     # no, ~0%
    'slide_cabinet_open_and_place_cups': 'cabinet', # no, ~0%
    'put_shoes_in_box':         'shoes',    # no, ~0%
    'open_jar':                 'jar',       # no, ~0%
    'put_tray_in_oven':         'tray',
    'plug_charger_in_power_supply': 'charger', # no, ~0%
    'set_the_table':            'table',     # no, ~0%
    'setup_checkers':           'checkers',
    'wipe_desk':                'desk',
    'remove_cups':              'remove', # 50before got better
    'put_money_in_safe':        'safe',  # no, ~10%
    'put_plate_in_colored_dish_rack':   'rack', 
    'stack_cups':               'scups',
    'put_item_in_drawer':       'pdrawer',
    'put_bottle_in_fridge':     'pfridge',
    'hannoi_square':            'hannoi',
    'take_frame_off_hanger':    'hanger', # 30%
    'memo.txt':                 'xx',
    'open_microwave':           'open_microwave', # ALWAYS errors out
    'put_knife_in_knife_block': 'put_knife_in_knife_block', # rlbench.task_environment.TaskEnvironmentError: Could not place the task put_knife_in_knife_block in the scene. This should not happen, please raise an issues on this task
    'pour_from_cup_to_cup': 'pour_from_cup_to_cup', #  Could not place the task put_knife_in_knife_block in the scene.
}

 
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
    os.makedirs(logdir, exist_ok=True)
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
        print('Initializing wandb run with keywords:', cfg.wandb)
        run = wandb.init(project='MTARM', **cfg.wandb)
        run.name = cfg.log_path
        cfg_dict = {}
        for key in ['rlbench', 'replay', 'framework', 'contexts']:
            for sub_key in cfg[key].keys():
                cfg_dict[key+'/'+sub_key] = cfg[key][sub_key]
        run.config.update(cfg_dict, allow_val_change=True)
        run.save()


    # trainer doesn't know if this is meta-rl agent 
    train_runner = PyTorchTrainRunner(
        agent, env_runner,
        wrapped_replays, device, replay_split, stat_accum,
        iterations=cfg.framework.training_iterations,
        log_freq=cfg.framework.log_freq, 
        logdir=logdir,
        weightsdir=weightsdir,
        replay_ratio=replay_ratio,
        transitions_before_train=cfg.framework.transitions_before_train,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        wandb_logging=cfg.framework.wandb_logging,
        save_freq=cfg.dev.save_freq,
        )
    
    
    train_runner.start(resume_dir)
    del train_runner
    del env_runner
    torch.cuda.empty_cache()
    wandb.finish() 


@hydra.main(config_name='config_multitask', config_path='conf')
def main(cfg: DictConfig) -> None: 
     
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

    
    tasks = sorted([ s.split('/')[-1] for s in  glob(join(DATA_DIR, '*')) ])
    tasks = [t for t in tasks if t not in SKIP_TASKS.keys() ]
    swept_tasks = [s.split('/')[-2] for s in glob(f'/home/mandi/ARM/log/*/*{cfg.framework.transitions_before_train}before*')]
    print('Skipping:', swept_tasks)
    tasks = [t for t in tasks if t not in swept_tasks ]

    cfg.rlbench.cameras = cfg.rlbench.cameras if isinstance(
        cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = _create_obs_config(cfg.rlbench.cameras,
                                    cfg.rlbench.camera_resolution) 
    
    cwd = os.getcwd() 

    seed = 99
    for task_name in tasks:
        log_path = join(cwd, task_name, cfg.method.name+'-'+cfg.run_name)
        logging.info('Logging to:' + log_path)
        cfg.log_path = log_path 
        logging.info('\n' + OmegaConf.to_yaml(cfg))

        os.makedirs(log_path, exist_ok=True)
        cfg.short_names = [task_name]
        cfg.tasks = [task_name]
        task_classes = [task_file_to_task_class(task_name)] 
        task_names = [task_name]

        env = CustomMultiTaskRLBenchEnv(
            task_classes=task_classes, task_names=task_names, observation_config=obs_config,
            action_mode=action_mode, dataset_root=cfg.rlbench.demo_path,
            episode_length=cfg.rlbench.episode_length, headless=True)


        logging.info('Starting seed %d.' % seed)
        run_seed(cfg, env, cfg.rlbench.cameras, device, seed, [task_name])


if __name__ == '__main__':
    main()

