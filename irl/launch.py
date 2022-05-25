"""
Launch agents with dense learned reward
"""
import os
import sys
sys.path.append('/shared/ademi_adeniji/ARM')
import pickle 
from arm.custom_rlbench_env_multitask import CustomMultiTaskRLBenchEnv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["QT_LOGGING_RULES" ] = '*.debug=false;qt.qpa.*=false'

import torch
from rlbench import ArmActionMode
from rlbench.action_modes import ActionMode, GripperActionMode
from rlbench.backend import task
from rlbench.backend.utils import task_file_to_task_class
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from custom_env_runners import EnvRunner
from custom_train_runner import PyTorchTrainRunner as TrainRunner
from custom_rollout_generator import CustomRolloutGenerator
from yarr.utils.stat_accumulator import MultiTaskAccumulatorV2

from arm import c2farm  
from arm.baselines import td3
import numpy as np
import hydra
import logging
from omegaconf import DictConfig, OmegaConf, ListConfig
from os.path import join 
import wandb 
from launch_multitask import _create_obs_config
from collections import defaultdict

ACTION_MODE = ActionMode(
        ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME,
        GripperActionMode.OPEN_AMOUNT)
LOG_CONFIG_KEYS = ['rlbench', 'replay', 'framework', 'rew', 'method', 'dev']

def _modify_action_min_max(action_min_max):
    # Make translation bounds a little bigger
    action_min_max[0][0:3] -= np.fabs(action_min_max[0][0:3]) * 0.2
    action_min_max[1][0:3] += np.fabs(action_min_max[1][0:3]) * 0.2
    action_min_max[0][-1] = 0
    action_min_max[1][-1] = 1
    action_min_max[0][3:7] = np.array([-1, -1, -1, 0])
    action_min_max[1][3:7] = np.array([1, 1, 1, 1])
    return action_min_max

def run_seed(
    cfg: DictConfig, 
    env, 
    cams, 
    device, 
    seed, 
    all_tasks, # [stack_blocks, push_buttons,...] 
    ) -> None:
    replay_path = join(cfg.replay.path, cfg.tasks_name, cfg.method.name, 'seed%d' % seed)
    action_min_max = None
    task_var_to_replay_idx = defaultdict(dict)
    if cfg.method.name == 'C2FARM':
        if cfg.replay.share_across_tasks:  
            logging.info(f'Using only one replay for multiple tasks, one batch size: {cfg.replay.batch_size}')
            r = c2farm.launch_utils.create_replay(
                cfg.replay.batch_size, 
                cfg.replay.timesteps,
                cfg.replay.prioritisation,
                replay_path if cfg.replay.use_disk else None, cams, env,
                cfg.method.voxel_sizes,
                replay_size=cfg.replay.replay_size
                ) 
            for i, (one_task, its_variations) in enumerate(zip(all_tasks, cfg.rlbench.use_variations)):
                for task_var in its_variations:
                    var = int( task_var.split("_")[-1])  
                    c2farm.launch_utils.fill_replay(
                        r, one_task, env, 
                        (cfg.rlbench.demos if one_task not in cfg.tasks.get('no_demo_tasks', []) else 0),
                        cfg.method.demo_augmentation, 
                        cfg.method.demo_augmentation_every_n,
                        cams, 
                        cfg.rlbench.scene_bounds,
                        cfg.method.voxel_sizes, 
                        cfg.method.bounds_offset,
                        cfg.method.rotation_resolution, 
                        cfg.method.crop_augmentation,
                        variation=var,
                        task_id=i,
                        )
                    task_var_to_replay_idx[i][var] = 0
                print(f"Task id {i}: {one_task}, **filled** replay for {len(its_variations)} variations")
            replays = [r]
        
        elif cfg.replay.share_across_vars:
            replays = []
            cfg.replay.replay_size = int(cfg.replay.replay_size / len(all_tasks))
            for i, (one_task, its_variations) in enumerate(zip(all_tasks, cfg.rlbench.use_variations)):
                r = c2farm.launch_utils.create_replay(
                        cfg.replay.batch_size, cfg.replay.timesteps, cfg.replay.prioritisation,
                        replay_path if cfg.replay.use_disk else None, 
                        cams, env,  cfg.method.voxel_sizes,  replay_size=cfg.replay.replay_size)
                for task_var in its_variations:
                    var = int( task_var.split("_")[-1])
                    c2farm.launch_utils.fill_replay(
                        r, one_task, env, 
                        (cfg.rlbench.demos if one_task not in cfg.tasks.get('no_demo_tasks', []) else 0),
                        cfg.method.demo_augmentation,  cfg.method.demo_augmentation_every_n,
                        cams, cfg.rlbench.scene_bounds, cfg.method.voxel_sizes,  cfg.method.bounds_offset, 
                        cfg.method.rotation_resolution, cfg.method.crop_augmentation,
                        augment_reward=False,
                        variation=var,
                        task_id=i
                        )
                    task_var_to_replay_idx[i][var] = len(replays) 
                replays.append(r)
        else:
            replays = []
            cfg.replay.replay_size = int(cfg.replay.replay_size / sum([len(_vars) for _vars in cfg.rlbench.use_variations]) )
            
            for i, (one_task, its_variations) in enumerate(zip(all_tasks, cfg.rlbench.use_variations)):
                for task_var in its_variations:
                    var = int( task_var.split("_")[-1]) 
                    r = c2farm.launch_utils.create_replay(
                        cfg.replay.batch_size, cfg.replay.timesteps, cfg.replay.prioritisation,
                        replay_path if cfg.replay.use_disk else None, 
                        cams, env,  cfg.method.voxel_sizes,  replay_size=cfg.replay.replay_size)
                    c2farm.launch_utils.fill_replay(
                        r, one_task, env, 
                        (cfg.rlbench.demos if one_task not in cfg.tasks.get('no_demo_tasks', []) else 0),
                        cfg.method.demo_augmentation,  cfg.method.demo_augmentation_every_n,
                        cams, cfg.rlbench.scene_bounds, cfg.method.voxel_sizes,  cfg.method.bounds_offset, 
                        cfg.method.rotation_resolution, cfg.method.crop_augmentation,
                        augment_reward=False,
                        variation=var,
                        task_id=i
                        )
                    task_var_to_replay_idx[i][var] = len(replays) 
                    replays.append(r)
                    
                # print(f"Task id {i}: {one_task}, **created** and filled replay for {len(its_variations)} variations")
            logging.info(f'Splitting total replay size into each buffer: {cfg.replay.replay_size}')
            print('Created mapping from var ids to buffer ids:', task_var_to_replay_idx)
            cfg.replay.total_batch_size = int(cfg.replay.batch_size * cfg.replay.buffers_per_batch)
            if cfg.dev.augment_batch > 0:
                cfg.replay.total_batch_size = int(
                    (cfg.replay.batch_size + cfg.dev.augment_batch) * cfg.replay.buffers_per_batch)
        
        agent = c2farm.launch_utils.create_agent(cfg, env) 
    elif cfg.method.name == 'TD3':
        action_min_max = (-1 * np.ones(8), np.ones(8))
        
        if cfg.replay.share_across_tasks:  
            logging.info(f'Using only one replay for multiple tasks, one batch size: {cfg.replay.batch_size}')
            r = td3.launch_utils.create_replay(
                cfg.replay.batch_size, cfg.replay.timesteps,
                cfg.replay.prioritisation,
                replay_path if cfg.replay.use_disk else None, env)
            for i, (one_task, its_variations) in enumerate(zip(all_tasks, cfg.rlbench.use_variations)):
                for task_var in its_variations:
                    var = int( task_var.split("_")[-1])  
                    task_var_to_replay_idx[i][var] = 0
                print(f"Task id {i}: {one_task}, **filled** replay for {len(its_variations)} variations")
            replays = [r]
        
        elif cfg.replay.share_across_vars:
            replays = []
            cfg.replay.replay_size = int(cfg.replay.replay_size / len(all_tasks))
            for i, (one_task, its_variations) in enumerate(zip(all_tasks, cfg.rlbench.use_variations)):
                r = td3.launch_utils.create_replay(
                    cfg.replay.batch_size, cfg.replay.timesteps,
                    cfg.replay.prioritisation,
                    replay_path if cfg.replay.use_disk else None, env)
                for task_var in its_variations:
                    var = int( task_var.split("_")[-1])
                    task_var_to_replay_idx[i][var] = len(replays) 
                replays.append(r)
        else:
            replays = []
            cfg.replay.replay_size = int(cfg.replay.replay_size / sum([len(_vars) for _vars in cfg.rlbench.use_variations]) )
            
            for i, (one_task, its_variations) in enumerate(zip(all_tasks, cfg.rlbench.use_variations)):
                for task_var in its_variations:
                    var = int( task_var.split("_")[-1]) 
                    r = td3.launch_utils.create_replay(
                        cfg.replay.batch_size, cfg.replay.timesteps,
                        cfg.replay.prioritisation,
                        replay_path if cfg.replay.use_disk else None, env)
                    task_var_to_replay_idx[i][var] = len(replays) 
                    replays.append(r)
                    
                # print(f"Task id {i}: {one_task}, **created** and filled replay for {len(its_variations)} variations")
            logging.info(f'Splitting total replay size into each buffer: {cfg.replay.replay_size}')
            print('Created mapping from var ids to buffer ids:', task_var_to_replay_idx)
            cfg.replay.total_batch_size = int(cfg.replay.batch_size * cfg.replay.buffers_per_batch)
            if cfg.dev.augment_batch > 0:
                cfg.replay.total_batch_size = int(
                    (cfg.replay.batch_size + cfg.dev.augment_batch) * cfg.replay.buffers_per_batch)
        
        agent = td3.launch_utils.create_agent(
            cams[0], cfg.method.activation, action_min_max,
            cfg.rlbench.camera_resolution, cfg.method.critic_lr,
            cfg.method.actor_lr, cfg.method.critic_weight_decay,
            cfg.method.actor_weight_decay, cfg.method.tau,
            cfg.method.critic_grad_clip, cfg.method.actor_grad_clip,
            env.low_dim_state_len)
    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    wrapped_replays = [PyTorchReplayBuffer(r, num_workers=1) for r in replays]
    # NOTE: stat accumulator still using task-based logging, too many variations
    # stat_accum = MultiTaskAccumulator(cfg.tasks, eval_video_fps=30) 
    logging.info('Creating Stat Accumulator: ')
    stat_accum = MultiTaskAccumulatorV2(
        task_names=all_tasks, 
        tasks_vars=cfg.rlbench.use_variations,
        eval_video_fps=30,
        mean_only=True,
        max_len=cfg.framework.num_log_episodes,
        log_all_vars=cfg.framework.log_all_vars,
        ) 

    logdir = join(cfg.log_path, 'seed%d' % seed)
    os.makedirs(logdir, exist_ok=False)
    OmegaConf.save( config=cfg, f=join(logdir, 'config.yaml') )
    weightsdir = join(logdir, 'weights')
    

    if action_min_max is not None:
        # Needed if we want to run the agent again
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, 'action_min_max.pkl'), 'wb') as f:
            pickle.dump(action_min_max, f) 

    num_all_vars = sum([len(variations) for variations in cfg.rlbench.use_variations])  

    rollout_generator = CustomRolloutGenerator( 
        one_hot=cfg.dev.one_hot, 
        noisy_one_hot=cfg.dev.noisy_one_hot, 
        num_task_vars=num_all_vars,
        task_var_to_replay_idx=task_var_to_replay_idx,
        replay_buffers=None,
        dev_cfg=cfg.dev,
        rew_cfg=cfg.rew,
        )

    device_list = [ i for i in range(torch.cuda.device_count()) ]
    assert len(device_list) > 1, 'Must use multiple GPUs'
    env_gpus = None 
    if len(device_list) > 1:
        print('Total visible GPUs idxed: ', device_list)
        env_gpus = device_list[cfg.framework.env_runner_gpu: ]
        print('Environment runner using GPUs idxed: ', env_gpus)

    env_runner = EnvRunner(
        train_env=env, 
        agent=agent, 
        train_replay_buffer=replays,
        rollout_generator=rollout_generator,
        num_train_envs=cfg.framework.train_envs,                
        num_eval_envs=cfg.framework.eval_envs,
        episodes=99999,
        episode_length=cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        max_fails=cfg.rlbench.max_fails,
        device_list=env_gpus, 
        task_var_to_replay_idx=task_var_to_replay_idx,
        eval_only=cfg.dev.eval_only, # only run eval EnvRunners 
        iter_eval=cfg.framework.ckpt_eval, # 
        eval_episodes=cfg.framework.num_log_episodes,
        log_freq=cfg.framework.log_freq, 
        target_replay_ratio=cfg.framework.replay_ratio,
        final_checkpoint_step=int(cfg.framework.training_iterations-1),
        dev_cfg=cfg.dev,
        rew_cfg=cfg.rew,
        )  

    if cfg.framework.wandb:
        run = wandb.init(**cfg.wandb)  
        run.name = "/".join( cfg.log_path.split('/')[-2:] )
        cfg_dict = {}
        for key in LOG_CONFIG_KEYS:
            for sub_key in cfg[key].keys():
                cfg_dict[key+'/'+sub_key] = cfg[key][sub_key]
        run.config.update(cfg_dict)
        run.save()
    
    resume_dir = None
    if cfg.resume:
        resume_dir = join(cfg.resume_path, cfg.resume_run, 'weights', str(cfg.resume_step))
        assert os.path.exists(resume_dir), 'Cannot find the weights saved at path: '+resume_dir
        cfg.framework.resume_dir = resume_dir 

    train_runner = TrainRunner(
        agent, env_runner,
        wrapped_replays, 
        device, 
        stat_accum,
        iterations=cfg.framework.training_iterations,
        eval_episodes=cfg.framework.eval_episodes,
        save_freq=cfg.framework.save_freq, 
        log_freq=cfg.framework.log_freq, 
        logdir=logdir,
        weightsdir=weightsdir,
        replay_ratio=cfg.framework.replay_ratio,
        transitions_before_train=cfg.framework.transitions_before_train,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        wandb_logging=cfg.framework.wandb,  
        one_hot=cfg.dev.one_hot,
        noisy_one_hot=cfg.dev.noisy_one_hot,
        num_vars=num_all_vars,
        buffers_per_batch=cfg.replay.buffers_per_batch,
        num_tasks_per_batch=cfg.replay.num_tasks_per_batch,
        update_buffer_prio=cfg.replay.update_buffer_prio,
        offline=cfg.dev.offline,
        eval_only=cfg.dev.eval_only,  
        switch_online_tasks=cfg.framework.switch_online_tasks,
        task_var_to_replay_idx=task_var_to_replay_idx,
        dev_cfg=cfg.dev,
        )
 
    if cfg.dev.eval_only:
        train_runner.evaluate(resume_dir)
    else:
        train_runner.start(resume_dir)
    del train_runner
    del env_runner
    torch.cuda.empty_cache()


@hydra.main(config_name='config_irl', config_path='/shared/ademi_adeniji/ARM/irl')
def main(cfg: DictConfig) -> None:  
    cfg.method = OmegaConf.load('/shared/ademi_adeniji/ARM/conf/method/TD3.yaml')
    if cfg.framework.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:%d" % cfg.framework.gpu) 
        torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    logging.info('Using device %s.' % str(device))

    tasks = sorted([t for t in cfg.tasks.all_tasks if t != cfg.tasks.heldout])
    task_classes = [task_file_to_task_class(t) for t in tasks]

    cfg.rlbench.cameras = cfg.rlbench.cameras if isinstance(
        cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = _create_obs_config(cfg.rlbench.cameras,
                                    cfg.rlbench.camera_resolution) 
        
    variation_idxs = [j for j in range(cfg.rlbench.num_vars)] if cfg.rlbench.num_vars > -1 else []
    if len(cfg.dev.handpick) > 0:
        logging.info('Hand-picking limited variation ids: ')
        print(cfg.dev.handpick)
        variation_idxs = cfg.dev.handpick

    logging.info(f'Creating Env with that samples only from below variations:')
    print(variation_idxs)

    env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes, task_names=tasks, observation_config=obs_config,
        action_mode=ACTION_MODE, dataset_root=cfg.rlbench.demo_path,
        episode_length=cfg.rlbench.episode_length, headless=True, 
        use_variations=variation_idxs, # the tasks may have different num of variations, deal w that later 
        )
    print('Done creating custom env')
    
    all_tasks = []
    var_count, use_vars_count = 0, 0
    all_variations = [] 
    use_variations = []
    for name, tsk in zip(tasks, task_classes):
        task = env.get_task(tsk) 
        count = task.variation_count()
        if name == 'put_groceries_in_cupboard':
            count = 6 
            print('put_groceries_in_cupboard has bugged variation6, skipping 6-9 ') 
        use_count = min(count, cfg.rlbench.num_vars) if cfg.rlbench.num_vars > -1 else count 
        all_tasks.append(name)
        var_count += count
        use_vars_count += use_count 
        all_variations.append([ f"{name}_{c}" for c in range(count) ])
        if len(cfg.dev.handpick) > 0:
            use_variations.append([ f"{name}_{c}" for c in cfg.dev.handpick ])
            logging.info('Hand-picked variation names: ') 
            print(use_variations)
        else:
            use_variations.append([ f"{name}_{c}" for c in range(use_count) ])
        #print(name, tsk ,all_variations) 
        logging.info(f"Task: {name}, a total of {count} variations avaliable, using {use_count} of them")
     
    # NOTE(Mandi) need to give a "blank" vanilla env so you don't get this error from env_runner.spinup_train_and_eval:
    # TypeError: can't pickle _thread.lock objects 
     
    env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes, task_names=tasks, observation_config=obs_config,
        action_mode=ACTION_MODE, dataset_root=cfg.rlbench.demo_path,
        episode_length=cfg.rlbench.episode_length, headless=True,
        use_variations=variation_idxs
        )
    
    cfg.rlbench.all_tasks = all_tasks
    if len(all_tasks) > 1:
        assert cfg.rew.use_r3m, 'CLIP reward doesnot support multi-task'
    cfg.rew.task_names = cfg.rlbench.all_tasks

    cfg.rlbench.id_to_tasks = [(i, tsk) for i, tsk in enumerate(all_tasks)]
    cfg.rlbench.all_variations = all_variations
    cfg.rlbench.use_variations = use_variations
    all_task_ids = [ i for i in range(len(all_tasks)) ]

    tasks_name = "-".join(tasks) + f"-{var_count}var"
    if len(tasks) > 2:
        tasks_name = f'{len(tasks)}Task-{use_vars_count}var'  
        logging.info(f'Got {len(tasks)} tasks, re-naming the run as: {tasks_name}')
    cfg.tasks_name = tasks_name 

    cwd = os.getcwd()

    cfg.run_name = cfg.run_name + f"-Replay_B{cfg.replay.batch_size}x{1 if cfg.replay.share_across_tasks else cfg.replay.buffers_per_batch}"
    log_path = join(cwd, tasks_name, cfg.run_name)
    os.makedirs(log_path, exist_ok=True) 
    existing_seeds = len(list(filter(lambda x: 'seed' in x, os.listdir(log_path))))
    logging.info('Logging to:' + log_path)
    cfg.log_path = log_path 
    # logging.info('\n' + OmegaConf.to_yaml(cfg))
 
    for seed in range(existing_seeds, existing_seeds + cfg.framework.seeds):
        logging.info('Starting seed %d.' % seed)
        run_seed(
            cfg, 
            env, 
            cfg.rlbench.cameras, 
            device, 
            seed, 
            all_tasks,  
            )



if __name__ == '__main__':
    main()

