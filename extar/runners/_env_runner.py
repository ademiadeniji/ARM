import copy
import logging
import os
import time
from multiprocessing import Process, Manager
from typing import Any, List, Union

import numpy as np
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
import torch 
import wandb 
from torch.multiprocessing import Manager, Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
from collections import defaultdict
# try:
#     if get_start_method() != 'spawn':
#         set_start_method('spawn', force=True)
# except RuntimeError:
#     pass

import cv2
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,3,1,1))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,3,1,1))

class _EnvRunner(object):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 episodes: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 weightsdir: str = None,
                 device_list: List[int] = None,
                 ):
        self._train_env = train_env
        self._eval_env = eval_env
        self._agent = agent
        self._train_envs = train_envs
        self._eval_envs = eval_envs
        self._episodes = episodes
        self._episode_length = episode_length
        self._rollout_generator = rollout_generator
        self._weightsdir = weightsdir
        self._previous_loaded_weight_folder = ''
        self._timesteps = timesteps
        self._p_args = {}
        self.p_failures = {}
        manager = Manager()
        self.write_lock = manager.Lock()
        self.stored_transitions = manager.list()
        self.stored_videos = manager.dict() 
        self.agent_summaries = manager.list()
        self._kill_signal = kill_signal
        self._step_signal = step_signal
        self._save_load_lock = save_load_lock
        self._current_replay_ratio = current_replay_ratio
        self._target_replay_ratio = target_replay_ratio
        #self._eval_device = torch.device("cpu") if eval_device is None else torch.device("cuda:%d" % eval_device)
        self._device_list, self._num_device = (None, 1) if device_list is None else (
            [torch.device("cuda:%d" % int(idx)) for idx in device_list], len(device_list))
        print('Internal EnvRunner is using GPUs:', self._device_list)
        #logging.info(f'Agent using GPU in the Env Runner?: {eval_device}')
        self._load_agent, self._load_step = None, None 

    def restart(self, name: str):
        p = Process(target=self._run_env, args=self._p_args[name], name=name)
        p.start()
        return p
        
    def restart_process(self, name: str):
        p = Process(target=self._run_env, args=self._p_args[name], name=name)
        p.start()
        return p

    def spin_up_envs(self, name: str, num_envs: int, eval: bool):
        ps = []
        for i in range(num_envs):
            n = name + str(i)
            self._p_args[n] = (n, eval)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start()
            ps.append(p)
        return ps

    def spinup_train_and_eval(self, n_train, n_eval, name='env'):
        ps = []
        for i in range(n_train):
            n = 'train_' + name + str(i)
            self._p_args[n] = (n, False, i)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start()
            ps.append(p)
        for j in range(i, i + n_eval):
            n = 'eval_' + name + str(j)
            self._p_args[n] = (n, True, j)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start()
            ps.append(p)
        return ps

    def _load_save(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # Only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        self.current_d = weight_folders[-1]
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # Rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def _run_env(self, name: str, eval: bool, proc_idx: int):

        self._name = name

        self._agent = copy.deepcopy(self._agent)
        proc_device = self._device_list[int(proc_idx % self._num_device)] if self._device_list is not None else None
        # print(f"Process index {proc_idx}, name {name}, using device:")
        # print(proc_device)
        self._agent.build(training=False, device=proc_device)
        # if self._load_agent is not None:
        #     logging.info(f'Loading online agent at step {self._load_step}')
        #     for qfunc, params in zip(self._agent._pose_agent._qattention_agents, params):
        #             qfunc._q.load_state_dict(params)
        #     print(self._agent._pose_agent._device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._train_env
        if eval:
            env = self._eval_env
        env.eval = eval
        env.launch()
        for ep in range(self._episodes):
            self._load_save()
            #print(f'Debugging: _EnvRunner eval {eval}, episode {ep}')
            
            logging.debug('%s: Starting episode %d.' % (name, ep))
            episode_rollout = []
            episode_video, video_tasks = [], []
            generator = self._rollout_generator.generator(
                self._step_signal, env, self._agent,
                self._episode_length, self._timesteps, eval)
            
            try:
                for replay_transition in generator:
                    while True:
                        if self._kill_signal.value:
                            env.shutdown()
                            return
                        if (eval or self._target_replay_ratio is None or
                                self._step_signal.value <= 0 or (
                                        self._current_replay_ratio.value >
                                        self._target_replay_ratio)):
                            break
                        time.sleep(1)
                        logging.debug(
                            'Agent. Waiting for replay_ratio %f to be more than %f' %
                            (self._current_replay_ratio.value, self._target_replay_ratio))

                    with self.write_lock:
                        if len(self.agent_summaries) == 0:
                            # Only store new summaries if the previous ones
                            # have been popped by the main env runner.
                            for s in self._agent.act_summaries():
                                self.agent_summaries.append(s)
                    episode_rollout.append(replay_transition)
                    #if env._record_cam is not None:
                        #print('Debugging: _EnvRunner', self._agent.act_summaries()[0].value.shape)
                    # episode_video.append( 
                    #     (self._agent.act_summaries(), 
                    #     (env._record_cam.capture_rgb() * 255).astype(np.uint8) ) )
                    #with self.write_lock:
                    episode_video.append( {k: v for k,v in replay_transition.info.items() if 'act_Qattention' in k or '_rgb' in k } )
                    video_tasks.append(replay_transition.info.get('task_name'))
                    
            except StopIteration as e:
                continue
            except Exception as e:
                env.shutdown()
                raise e

            with self.write_lock:
                for transition in episode_rollout:
                    self.stored_transitions.append((name, transition, eval))
                    #if transition.terminal:
                    #    print('Debugging: _EnvRunner', transition.info['task_name'], transition.reward)
                self.stored_videos[video_tasks[0]] = episode_video
                img_arrays = defaultdict(list)
                
                if len(episode_video) > 1:
                    wandb.init(project='rlbench', job_type='debug')
                    for step, transition in enumerate(episode_rollout):
                        tsk = transition.info.get('task_name')
                        for k, img in transition.info.items():
                            if '_rgb' in k: 
                                if len(img.shape) == 5:
                                    img = img[0,0]
                                if len(img.shape) == 4:
                                    img = img[0]
                                if img.max() == 1:
                                    img = ((img + 1)/2  * 255).astype(np.uint8)
                                wandb.log({ tsk + '_' + k + 'step_' + str(step): wandb.Image(img)})

                    
                    for vid_step, tsk in zip(episode_video, video_tasks):
                        #print([ (k, v.shape) for k, v in vid_step.items()])
                        print(f"Task: {tsk}")
                        for k, v in vid_step.items():
                            img = v.transpose(2,0,1) if (v.shape[-1] == 3 and len(v.shape) == 3 ) else v 
                            if len(img.shape) == 5:
                                img = img[0,0]
                            if len(img.shape) == 4:
                                img = img[0]
                            if img.max() == 1:
                                img = ((img + 1)/2  * 255).astype(np.uint8)
                            
                            img_arrays[k].append(img) 
                    for k, v in img_arrays.items():
                        print(f'logging video {k}, length {len(v)}, {v[0].min(),v[0].max() }')
                        stacked = np.stack(v) 
                        wandb.log({video_tasks[0] + '_' + k: wandb.Video( stacked , fps=1)})
                    wandb.finish()
                    raise ValueError
 
                
        env.shutdown()

    def kill(self):
        self._kill_signal.value = True

    def update_failures(self, p, max_fails):
        assert p.name in self.p_failures.keys(), f'Process {p.name} not found in current runner'
        self.p_failures[p.name] += 1
        n_failures = self.p_failures[p.name]
        if n_failures > max_fails:
            logging.error('Process %s failed too many times (%d times > %d)' %
                        (p.name, n_failures, max_fails))
            raise RuntimeError('Too many process failures.')
        logging.warning('Env %s failed (%d times <= %d). restarting' %
                        (p.name, n_failures, max_fails))
        return self.restart(p.name)
