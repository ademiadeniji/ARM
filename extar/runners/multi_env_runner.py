"""
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

import numpy as np
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners._env_runner import _EnvRunner
from arm.custom_rlbench_env import CustomRLBenchEnv, MultiTaskRLBenchEnv
from yarr.utils.stat_accumulator import StatAccumulator
from yarr.agents.agent import Summary, ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary
from extar.utils.logger import MultiTaskAccumulator, MultiTaskAccumulator
from extar.utils.rollouts import MultiTaskRolloutGenerator

NUM_WEIGHTS_TO_KEEP = 10

class MultiTaskEnvRunner(EnvRunner):
    """Inherits Stephen's EnvRunner to update to multiple replay buffers"""
    def __init__(self, 
                env: MultiTaskRLBenchEnv, 
                agent: Agent,
                replays, #OrderedDict[ReplayBuffer],
                n_train: int, 
                n_eval: int,
                episodes: int,
                episode_length: int,
                stat_accumulator: Union[MultiTaskAccumulator, None] = None,
                rollout_generator: RolloutGenerator = None,
                weightsdir: str = None,
                max_fails: int = 5):

        super(MultiTaskEnvRunner, self).__init__(
            env, agent, list(replays.values())[0], # needs overwrite
            n_train, n_eval, episodes, episode_length, 
            stat_accumulator, rollout_generator, weightsdir, max_fails)
        # overwrite some stuff
        self._replays = replays
        #self._replay_buffer =  # for 'timesteps'
        self.n_tasks = len(replays)

        self._new_transitions = {}
        for task_name in env.unique_tasks.keys():
            self._new_transitions[task_name+'_train'] = 0 
            self._new_transitions[task_name+'_eval'] = 0
         
        self._total_transitions = deepcopy(self._new_transitions)
        self.last_step_time  = 0
        print(f'Using {self.n_tasks} replay buffers for tasks:', replays.keys() ) 
    

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
    
    def set_step(self, step):
        self._step_signal.value = step
        self.last_step_time = time.time()
        
    def time_since_last_step(self):
        return time.time() - self.last_step_time 

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
 

# class MultiEnvRunner(object):
#     """Manages a list of base _EnvRunners, each uses a different RLBench task"""
#     def __init__(
#         self, 
#         envs: MultiTaskRLBenchEnv,
#         agent: Agent,
#         replays: OrderedDict(ReplayBuffer),
#         n_train: int, 
#         n_eval: int,
#         episodes: int,
#         episode_length: int,
#         stat_accumulator: Union[StatAccumulator, None] = None,
#         rollout_generator: RolloutGenerator = None,
#         weightsdir: str = None,
#         max_fails: int = 10,
#         ):
#         self._envs = envs 
#         self._agent = agent
#         self.n_train, self.n_eval = n_train, n_eval 
#         self.episodes = episodes 
#         self.episode_length = episode_length 
#         self._stat_accumulator = {task: stat_accumulator for task in self._envs.get_unique_tasks()}
#         self._new_transitions = {
#             task: {'train_envs': 0, 'eval_envs': 0} for task in self._envs.get_unique_tasks()}
#         self._total_transitions = copy.deepcopy(self._new_transitions)

#         self._rollout_generator = (
#             RolloutGenerator() if rollout_generator is None
#             else rollout_generator)
#         self._weightsdir = weightsdir
#         self._max_fails = max_fails
#         #self.prefix_list = [f'_{name}' for name in envs.task_names]
#         self._kill_signal = Value('b', 0)
#         self._step_signal = Value('i', -1)
#         self._agent_summaries = defaultdict(list)
#         assert len(replays) == envs.n_train_tasks, 'Must have one replay buffer for each training task'
#         self._replays = replays 
#         self.target_replay_ratio = None  # Will get overridden later
#         self.current_replay_ratio = Value('f', -1)

#     def _multi_env_run(sef, save_load_lock):
#         # TODO(Mandi): add more logic to step only a subset of tasks
#         self._internal_runners = OrderedDict() # let's use only one runner for each task, but can choose whether to use train/eval env inside the one runne
#         for task in self._envs.get_unique_tasks():
#             timesteps = self._replays[task].timesteps
#             self._internal_runners[task] = _EnvRunner(
#                 train_env=self._envs.get_task_env(task), 
#                 eval_env=self._envs.get_task_env(task), 
#                 agent=self._agent, 
#                 timesteps=timesteps, 
#                 train_envs=self.n_train, eval_envs=self.n_eval,
#                 episodes=self._episodes, episode_length=self._episode_length, 
#                 kill_signal=self._kill_signal, step_signal=self._step_signal, 
#                 rollout_generator=self._rollout_generator, 
#                 save_load_lock=save_load_lock,
#                 current_replay_ratio=self.current_replay_ratio, 
#                 target_replay_ratio=self.target_replay_ratio,
#                 weightsdir=self._weightsdir)
        
#         all_env_procs = OrderedDict()
#         for mode, n in zip(['train', 'eval'], [self.n_train, self.n_eval]):
#             for task in self._envs.task_names[mode]:
#                 procs = self._internal_runners[task].spin_up_envs(
#                     name=task+f'_{mode}', num_envs=n, eval=(mode=='eval'))
#                 all_env_procs[task+f'_{mode}'] = (task, procs)
                 
#         no_transitions = {name: 0 for name in all_env_procs.keys()}
#         while True:
#             for name, (task, procs) in all_env_procs.items():
#                 for p in procs:
#                     if p.exitcode is not None:
#                         procs.remove(p)
#                         if p.exitcode != 0:
#                             new_p = self._internal_runners[task].update_failures(p, self._max_fails) 
#                             # handle warning inside the base runner
#                             procs.append(p)

#             if not self._kill_signal.value:
#                 new_transitions = self._update()
#                 for name, (task, procs) in all_env_procs.items():
#                     for p in procs:
#                         no_transitions[p.name] = 1 + no_transitions[p.name] if new_transitions[p.name] == 0 else 0 
#                         if no_transitions[p.name] > 600:  
#                             logging.warning("Env %s hangs, so restarting" % p.name)
#                             procs.remove(p)
#                             os.kill(p.pid, signal.SIGTERM)
#                             p = self._internal_runners[task].restart_process(p.name)
#                             procs.append(p)
#                             no_transitions[p.name] = 0
#             if len(all_env_procs) == 0:
#                 break
#             time.sleep(1)
        
#     def _update(self): # do explicit task-specific replay update
#         new_transitions = defaultdict(int)
#         for task, runner in self._internal_runners.items():
#             with runner.write_lock:
#                 self.agent_summaries[task] = list(runner.agent_summaries)
#                 if self._step_signal.value % self.log_freq == 0 and self._step_signal.value > 0:
#                     runner.agent_summaries[:] = []
#                 for name, transition, eval in runner.stored_transitions:
#                     if not eval:
#                         kwargs = dict(transition.observation)
#                         self._replays[task].add( 
#                             np.array(transition.action), transition.reward,
#                             transition.terminal, transition.timeout, **kwargs)
#                         if transition.terminal:
#                             self._replays[task].add_final(**transition.final_observation)
#                     new_transitions[name] += 1
#                     self._new_transitions[task]['eval_envs' if eval else 'train_envs'] += 1
#                     self._total_transitions[task]['eval_envs' if eval else 'train_envs'] += 1
#                     if self._stat_accumulator[task] is not None:
#                         self._stat_accumulator[task].step(transition, eval)
#                 runner.stored_transitions[:] = []
#         return new_transitions
    
#     def start(self, save_load_lock):
#         self._p = Thread(target=self._multi_env_run, args=(save_load_lock,), daemon=True)
#         self._p.name = 'MultiEnvRunnerThread'
#         self._p.start()

#     def wait(self):
#         if self._p.is_alive():
#             self._p.join()

#     def stop(self):
#         if self._p.is_alive():
#             self._kill_signal.value = True
#             self._p.join()

#     def set_step(self, step):
#         self._step_signal.value = step
    
#     def get_task_names(self, train=True):
#         return self._env.task_names['train'] if train else self._env.task_names['eval']
 
