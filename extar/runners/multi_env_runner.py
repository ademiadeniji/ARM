"""
New(0721) try adding GPU agents
NOTE(0714) My attempt at making a simpler version of a multi-task trainer that 
handles EnvRunner, ReplayBuffer, and the Agent Update with a slightly more
flattened logic.
Big credit to Stephen's MultiTaskQAttention for great reference 
"""
import os
import time
import copy
import wandb 
import torch
import logging
from copy import deepcopy 
from collections import OrderedDict, defaultdict
from multiprocessing import Process, Manager, Value
from typing import Any, List, Union

import collections
import logging
import os
import signal
import time
 
from threading import Thread 
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.replay_buffer.replay_buffer import ReplayBuffer
 
import numpy as np
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.envs.env import Env
# from yarr.utils.rollout_generator import RolloutGenerator
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.runners.env_runner import EnvRunner

from arm.custom_rlbench_env import CustomRLBenchEnv, MultiTaskRLBenchEnv
from yarr.utils.stat_accumulator import StatAccumulator
from yarr.agents.agent import Summary, ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary

from extar.utils.logger import MultiTaskAccumulator, MultiTaskAccumulator
from extar.utils.rollouts import RolloutGenerator
from extar.runners._env_runner import _EnvRunner # New(0724)

NUM_WEIGHTS_TO_KEEP = 10

class MultiTaskEnvRunner(object):
    """Inherits Stephen's EnvRunner to update to multiple replay buffers"""
    def __init__(self, 
                env: MultiTaskRLBenchEnv, 
                agent: Agent,
                replays, #OrderedDict[ReplayBuffer],
                device_list, # List[int] = None,
                n_train: int, 
                n_eval: int,
                episodes: int,
                episode_length: int,
                stat_accumulator: Union[MultiTaskAccumulator, None] = None,
                rollout_generator: RolloutGenerator = None,
                weightsdir: str = None,
                max_fails: int = 5,
                use_gpu: bool = False,
                receive: bool = False
                ):
        self._env = env 
        self._agent = agent 
        self._n_train, self._n_eval = n_train, n_eval
        # overwrite some stuff
        self._replays = replays
        
        self.n_tasks = len(replays)
        logging.info(f'Using {self.n_tasks} replay buffers only for training tasks:')
        print(replays.keys() )
        self._timesteps = list(replays.values())[0].timesteps 
        self._episodes = episodes
        self._episode_length = episode_length
        self._stat_accumulator = stat_accumulator
        self._rollout_generator = (RolloutGenerator() if rollout_generator is None else rollout_generator)
        self._weightsdir = weightsdir
        self._max_fails = max_fails
        self._previous_loaded_weight_folder = ''
        self._p = None
        self._kill_signal = Value('b', 0)
        self._step_signal = Value('i', -1)
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        self._total_transitions = {'train_envs': 0, 'eval_envs': 0}
        self.log_freq = 1000  # Will get overridden later
        self.target_replay_ratio = None  # Will get overridden later
        self.current_replay_ratio = Value('f', -1)

        self._new_transitions = {}
        for task_name in env.unique_tasks.keys():
            self._new_transitions[task_name+'_train'] = 0 
            self._new_transitions[task_name+'_eval'] = 0
        self._total_transitions = deepcopy(self._new_transitions)
        self._last_step_time  = Value('i', 0)
        
        self.use_gpu = use_gpu # for eval 
        if use_gpu:
            logging.info('Using GPU device idx %s for agents loaded in EnvRunner' % device_list)
        self.device_list = device_list 
        self._receive = receive
        self.curr_agent = None
        self.curr_train_step = 0
   
    def _update(self):
        """Only difference is routing transitions to different replays, internal runner doesnt need to organize """
        new_transitions = defaultdict(int)
        with self._internal_env_runner.write_lock:
            self._agent_summaries = list(
                self._internal_env_runner.agent_summaries)
            if self._step_signal.value % self.log_freq == 0 and self._step_signal.value > 0:
                self._internal_env_runner.agent_summaries[:] = []
            for name, transition, eval in self._internal_env_runner.stored_transitions:
                task_name = transition.info.get('task_name', None)
                assert task_name, 'Multi-task transitions must always store which task this is'
                
                if not eval:
                    kwargs = dict(transition.observation) 
                    replay = self._replays.get(task_name, None)
                    assert replay, f'Cannot match the task {task_name} with a replay buffer'
                    replay.add(
                        np.array(transition.action), transition.reward,
                        transition.terminal,
                        transition.timeout, **kwargs)
                    if transition.terminal:
                        replay.add_final(
                            **transition.final_observation)
                new_transitions[name] += 1
                #assert task_name not in self._new_transitions.keys() 
                self._new_transitions[task_name+'_eval' if eval else task_name+'_train'] += 1
                self._total_transitions[task_name+'_eval' if eval else task_name+'_train'] += 1
                if self._stat_accumulator is not None:
                    self._stat_accumulator.step(transition, eval)
            self._internal_env_runner.stored_transitions[:] = []  # Clear list
        return new_transitions
 
    def receive(self, incoming, train_step):
        self.curr_agent = incoming 
        self.curr_train_step = train_step 

    def _run(self, save_load_lock):
        """Give internal runner a eval gpu """
        self._internal_env_runner = _EnvRunner(
            self._env, self._env, self._agent, self._timesteps, 
            self._episodes, self._episode_length, 
            self._kill_signal, self._step_signal, 
            self._rollout_generator, 
            save_load_lock,
            self.current_replay_ratio, 
            self.target_replay_ratio,
            self._weightsdir, 
            device_list=self.device_list if self.use_gpu else None, 
            )

        envs = self._internal_env_runner.spinup_train_and_eval(n_train=self._n_train, n_eval=self._n_eval, name='_env')
        # training_envs = self._internal_env_runner.spin_up_envs('train_env', self._n_train, False)
        # eval_envs = self._internal_env_runner.spin_up_envs('eval_env', self._n_eval, True)
        # envs = training_envs + eval_envs
        no_transitions = {env.name: 0 for env in envs}
        while True:
            for p in envs:
                if p.exitcode is not None:
                    envs.remove(p)
                    if p.exitcode != 0:
                        self._internal_env_runner.p_failures[p.name] += 1
                        n_failures = self._internal_env_runner.p_failures[p.name]
                        if n_failures > self._max_fails:
                            logging.error('Env %s failed too many times (%d times > %d)' %
                                          (p.name, n_failures, self._max_fails))
                            raise RuntimeError('Too many process failures.')
                        logging.warning('Env %s failed (%d times <= %d). restarting' %
                                        (p.name, n_failures, self._max_fails))
                        p = self._internal_env_runner.restart(p.name)
                        envs.append(p)

            if not self._kill_signal.value:
                new_transitions = self._update()
                for p in envs:
                    if new_transitions[p.name] == 0:
                        no_transitions[p.name] += 1
                    else:
                        no_transitions[p.name] = 0
                    if no_transitions[p.name] > 600:  # 5min
                        logging.warning("Env %s hangs, so restarting" % p.name)
                        envs.remove(p)
                        os.kill(p.pid, signal.SIGTERM)
                        p = self._internal_env_runner.restart_process(p.name)
                        envs.append(p)
                        no_transitions[p.name] = 0

            if len(envs) == 0:
                break
            time.sleep(1)
    
    def summaries(self): # -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop())
        for key, value in self._new_transitions.items():
            summaries.append(ScalarSummary('%s/new_transitions' % key, value))
        for key, value in self._total_transitions.items():
            summaries.append(ScalarSummary('%s/total_transitions' % key, value))
        for k in self._new_transitions.keys():
            self._new_transitions[k] = 0
        summaries.extend(self._agent_summaries)
        return summaries
    
    def start(self, save_load_lock):
        self._p = Thread(target=self._run, args=(save_load_lock,), daemon=True)
        self._p.name = 'EnvRunnerThread'
        self._p.start()

    def wait(self):
        if self._p.is_alive():
            self._p.join()

    def stop(self):
        if self._p.is_alive():
            self._kill_signal.value = True
            self._p.join()
    
    def set_step(self, step):
        self._step_signal.value = step
        self._last_step_time.value = time.time()
        
    def time_since_last_step(self):
        return time.time() - self._last_step_time.value  
