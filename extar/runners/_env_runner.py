"""Change based on yarr.runners._env_runner to choose between multiple GPUs for agent evaluation """
import copy
from copy import deepcopy 
import logging
import os
import time
import multiprocessing
from multiprocessing import Process, Manager

from typing import Any, List

import numpy as np
from yarr.agents.agent import Agent
from yarr.envs.env import Env
# from yarr.utils.rollout_generator import RolloutGenerator
from extar.utils.rollouts import RolloutGenerator
import torch 

class _EnvRunner(object):
    
    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
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
                 receive=False,
                 incoming=None, 
                 train_step=None,
                 ):
        self._train_env = train_env
        self._eval_env = eval_env 
        self._agent = agent
        self._agent_step = 0
        self.receive = receive
        self.waiting = incoming 
        self.curr_train_step = train_step 
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
        self.agent_summaries = manager.list()
        self._kill_signal = kill_signal
        self._step_signal = step_signal
        self._save_load_lock = save_load_lock
        self._current_replay_ratio = current_replay_ratio
        self._target_replay_ratio = target_replay_ratio
        self._device_list = torch.device("cpu") if device_list is None else [torch.device("cuda:%d" % idx) for idx in device_list]
        self._n_device = None if device_list is None else len(device_list)
        if self._n_device is not None:
            logging.info(f'_EnvRunner is splitting {self._n_device} devices with index {device_list}')
        else:
            logging.info('Warning! Agent NOT using GPU in internal _EnvRunner')

    # def restart_process(self, name: str):
    #     p = Process(target=self._run_env, args=self._p_args[name], name=name)
    #     p.start()
    #     return p where the fuck did this come from??
    def restart(self, name: str):
        p = Process(target=self._run_env, args=self._p_args[name], name=name)
        p.start()
        return p

    def spin_up_envs(self, name: str, num_envs: int, eval: bool):
        ps = []
        for i in range(num_envs):
            n = name + str(i)
            self._p_args[n] = (n, eval, i)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start()
            ps.append(p)
        return ps
    
    def spinup_train_and_eval(self, n_train: int, n_eval: int, name: str = '_env'):
        # add logic to split devices
        ps = []
        for i in range(n_train):
            proc_name = 'train' + name + str(i)
            self._p_args[proc_name] = (proc_name, False, i)
            self.p_failures[proc_name] = 0
            p = Process(target=self._run_env, args=self._p_args[proc_name], name=proc_name)
            p.start() 
            ps.append(p)
        
        for j in range(i, i + n_eval):
            proc_name = 'eval' + name + str(j)
            self._p_args[proc_name] = (proc_name, True, j)
            self.p_failures[proc_name] = 0
            p = Process(target=self._run_env, args=self._p_args[proc_name], name=proc_name)
            p.start() 
            ps.append(p)

        return ps 

    def _load_save(self, device=None):
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

    def receive_online_agent(self, incoming, train_step, device=None):
        self._agent.load_agent(incoming)
        logging.info('Debugging: the device agent got loaded to v.s. the device this runner process Should use:', self._agent._device)
        self._agent_step = train_step 
        logging.info('Debugging: agent step vs step signal inside envrunner:', train_step, self._step_signal.value )

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def _run_env(self, name: str, eval: bool, proc_idx: int):

        self._name = name
 

        self._agent = copy.deepcopy(self._agent)

        eval_device = None if self._n_device is None else self._device_list[ int(proc_idx % self._n_device) ]
        #self._curr_device = eval_device
        self._agent.build(training=False, device=eval_device)
        if self.receive:
            self.receive_online_agent(self.waiting, self.curr_train_step)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env if eval else self._train_env
        env.eval = eval
        env.launch()
        for ep in range(self._episodes):
            self._load_save()
            logging.debug('%s: Starting episode %d.' % (name, ep))
            episode_rollout = []
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
            except StopIteration as e:
                continue
            except Exception as e:
                env.shutdown()
                raise e

            with self.write_lock:
                for transition in episode_rollout:
                    self.stored_transitions.append((name, transition, eval))
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

