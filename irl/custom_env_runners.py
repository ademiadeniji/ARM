import collections
import logging
import os
import signal
import time
from multiprocessing import Value, Manager
from threading import Thread
from typing import List
from typing import Union
import random
import numpy as np

from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary, ImageSummary, VideoSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import StatAccumulator
from glob import glob 
import torch 
from copy import deepcopy 
import pickle 
TASK_ID='task_id'
VAR_ID='variation_id'
WAIT_TIME=2000 # original version was 600 -> 5min
""" Note how episode rollout videos are logged/generated: custom_rlbench_env_multitask records episodes,
whenever terminal, returns as 'summaries' in act_result and send to multitask_rollout_generator, then 
stay as part of _EnvRunner.stored_transitions and get returned to stat_accumulator by EnvRunner  """
import copy
import logging
import os
import time
from multiprocessing import Value, Process, Manager
from typing import Any, List, Union 
 
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from multiprocessing import get_start_method, set_start_method


try:
    if get_start_method() != 'spawn':
        set_start_method('spawn', force=True)
except RuntimeError:
    pass
 
WAIT_WARN=200
CHECKPT='agent_checkpoint'

class _EnvRunner(object):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 episodes: int,
                 eval_episodes: int, 
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 online_task_ids, # limit the task options for env runner 
                 weightsdir: str = None,
                 device_list: List[int] = None, 
                 all_task_var_ids = None,
                 final_checkpoint_step = 999, 
                 ):
        self._train_env = train_env
        self._eval_env = eval_env
        self._agent = agent
        self._train_envs = train_envs
        self._eval_envs = eval_envs
        self._episodes = episodes
        self._eval_episodes = eval_episodes # evaluate each agent checkpoint this num eps for each task variation 
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
        self.clip_episodes = manager.list()

        self.stored_ckpt_eval_transitions = manager.dict() 
        self.agent_ckpt_eval_summaries = manager.dict()

        self._kill_signal = kill_signal
        self._step_signal = step_signal

        self._agent_checkpoint = Value('i', -1)
        self._loaded_eval_checkpoint = manager.list()
        self._finished_eval_checkpoint = manager.list()
        self.final_checkpoint_step = final_checkpoint_step
        logging.info('Expected final checkpoint to be saved at step: {}'.format(self.final_checkpoint_step))

        self._save_load_lock = save_load_lock
        self._current_replay_ratio = current_replay_ratio
        self._target_replay_ratio = target_replay_ratio
        self.online_task_ids = online_task_ids 
        self._all_task_var_ids = all_task_var_ids

        self._device_list, self._num_device = (None, 1) if device_list is None else (
            [torch.device("cuda:%d" % int(idx)) for idx in device_list], len(device_list))
        print('Internal EnvRunner is using GPUs:', self._device_list)
        self.online_buff_id = Value('i', -1)
  
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
    
    def spinup_train_and_eval(self, n_train, n_eval, name='env', iter_eval=False):
        ps = []
        i = 0
        # num_cpus = os.cpu_count()
        # print(f"Found {num_cpus} cpus, limiting:")
        # per_proc = int(num_cpus / (n_train+n_eval))
        for i in range(n_train):
            n = 'train_' + name + str(i)
            self._p_args[n] = (n, False, i)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start() 
            # print(os.system(f"taskset -cp {int(i * per_proc)}-{int( (i+1) * per_proc )} {p.pid}" ))
            ps.append(p)
        
        for j in range(n_train, n_train + n_eval):
            n = 'eval_' + name + str(j)
            self._p_args[n] = (n, True, j)
            self.p_failures[n] = 0
            p = Process(target=(self._iterate_all_vars if iter_eval else self._run_env), args=self._p_args[n], name=n)
            p.start()
            # print(os.system(f"taskset -cp {int(j * per_proc)}-{min(num_cpus-1, int( (j+1) * per_proc )) } {p.pid}" ))
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
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # Rare case when agent hasn't finished writing.
                            logging.warning('_EnvRunner: agent hasnt finished writing.')
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _load_next_unevaled(self):
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
                    new_weight_folders = [w for w in weight_folders if w not in self._loaded_eval_checkpoint ]
                    # load the next unevaluated checkpoint  
                if len(new_weight_folders) > 0:
                    w = new_weight_folders[0]  
                    self._agent_checkpoint.value = int(w)
                    d = os.path.join(self._weightsdir, str(w))
                    try:
                        self._agent.load_weights(d)
                    except FileNotFoundError:
                        # Rare case when agent hasn't finished writing.
                        logging.warning('_EnvRunner: agent hasnt finished writing.')
                        time.sleep(1)
                        self._agent.load_weights(d)
                    print('Agent %s: Loaded weights: %s for evaluation' % (self._name, d)) 
                    with self.write_lock:
                        self._loaded_eval_checkpoint.append(w) 
                    return False 
            logging.info('Waiting for weights to become available.') 
            if max(self._loaded_eval_checkpoint) == self.final_checkpoint_step:
                print('Found %s final checkpoint, Stop looking for new saved checkpoints' % self._name)
                return int(self._agent.get_checkpoint()) == -1
            time.sleep(1)  

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def _run_env(self, name: str, eval: bool, proc_idx: int):
        
        self._name = name

        self._agent = copy.deepcopy(self._agent)
        
        proc_device = self._device_list[int(proc_idx % self._num_device)] if self._device_list is not None else None
        self._agent.build(training=False, device=proc_device)

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
            logging.debug('%s: Starting episode %d.' % (name, ep)) 
            if not eval and len(self.online_task_ids) > 0:
                logging.debug(f"env runner setting online tasks: {self.online_task_ids}")
                #print("Setting avaliable tasks", self.online_task_ids)
                env.set_avaliable_tasks(self.online_task_ids) 
            if not eval and self.online_buff_id.value > -1:
                #print(f'online buffer: {self.online_buff_id.value}')
                task_id, var_id = random.choice(
                    self.replay_to_task_vars[self.online_buff_id.value]
                    )
                #print(f'online : task_id: {task_id}, var_id: {var_id}')
                env.set_task_variation(task_id, var_id) 
            episode_rollout = []
            generator = self._rollout_generator.generator(
                self._step_signal, env, self._agent,
                self._episode_length, self._timesteps, eval, 
                swap_task=(False if not eval and self.online_buff_id.value > -1 else True), 
                )
            try:
                for replay_transition, episode_success in generator:
                    slept = 0
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
                        slept += 1
                        logging.info(
                            'Agent. Waiting for replay_ratio %f to be more than %f' %
                            (self._current_replay_ratio.value, self._target_replay_ratio))

                        if slept % WAIT_WARN == 0:
                            logging.warning(
                            'Env Runner process %s have been waiting for replay_ratio %f to be more than %f for %d seconds' %
                            (name, self._current_replay_ratio.value, self._target_replay_ratio, slept))
                            
                    with self.write_lock:
                        # logging.warning(f'proc {name}, idx {proc_idx} writing agent summaries')
                        if len(self.agent_summaries) == 0:
                            # Only store new summaries if the previous ones
                            # have been popped by the main env runner.
                            for s in self._agent.act_summaries():
                                self.agent_summaries.append(s)
                        # logging.warning(f'proc {name}, idx {proc_idx} finished writing agent summaries')
                    
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                env.shutdown()
                raise e

            with self.write_lock: 
                # logging.warning(f'proc {name}, idx {proc_idx} adding to stored transitions')
                if self._rollout_generator._rew_cfg.save_data: 
                    self.clip_episodes.append(episode_rollout)
                for transition in episode_rollout:
                    if 'task_success' in transition.info.keys():
                        transition.info.pop('task_success')
                    self.stored_transitions.append((name, transition, eval))
                
                # logging.warning(f'proc {name}, idx {proc_idx} finished adding to stored transitions') 
        env.shutdown()

    def _iterate_all_vars(self, name: str,  eval: bool, proc_idx: int): 
        # use for eval env only 
        self._name = name
        self._agent = copy.deepcopy(self._agent)
        proc_device = self._device_list[int(proc_idx % self._num_device)] if self._device_list is not None else None
        self._agent.build(training=False, device=proc_device)
        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)
        ckpt = None 
        env = self._eval_env
        env.eval = True 
        env.launch()
         
        while True: 
            shutdown = self._load_next_unevaled()
            if shutdown:
                print('Shutting down process %s since other threads are evaluating the last checkpoint' % name)
                env.shutdown()
                return 
            # if self._kill_signal.value and ckpt is not None and ckpt == int(self._agent.get_checkpoint()) :
            #     logging.info('No new checkpoint got loaded, shutting down')  
            #     env.shutdown()
            #     return 
            ckpt = int(self._agent.get_checkpoint()) 
            assert ckpt not in self.stored_ckpt_eval_transitions.keys(), 'There should be no transitions stored for this ckpt'
            # with self.write_lock:
            #     self.stored_ckpt_eval_transitions[ckpt] = []
            #     self.agent_ckpt_eval_summaries[ckpt] = [] 
            all_episode_rollout = []
            all_agent_summaries = []
            num_eval_steps = 0
            for (task_id, var_id) in self._all_task_var_ids:
                
                if self._kill_signal.value: 
                    logging.info('Finishing evaluation before full shutdown process', name, 'evaluating task + var:', task_id, var_id)
                #print('Setting task + var:', task_id, var_id)
                env.set_task_variation(task_id, var_id)
                for ep in range(self._eval_episodes):
                    # print('%s: Starting episode %d.' % (name, ep))
                    episode_rollout = []
                    generator = self._rollout_generator.generator(
                        self._step_signal, env, self._agent,
                        self._episode_length, self._timesteps, True, 
                        swap_task=False, 
                        )
                    try:
                        for replay_transition in generator:    
                            for s in self._agent.act_summaries():
                                s.step = ckpt
                                all_agent_summaries.append(s)
                                # logging.warning(f'proc {name}, idx {proc_idx} finished writing agent summaries')
                            assert replay_transition.info[CHECKPT] == ckpt, 'Checkpoint mismatch between transition in rollout and agent loaded point'
                            if 'task_success' in replay_transition.info.keys():
                                replay_transition.info.pop('task_success')

                            episode_rollout.append(replay_transition)
                            # print(replay_transition.info, env._task._variation_number )
                    except StopIteration as e:
                        continue
                    except Exception as e:
                        env.shutdown()
                        raise e

                    for transition in episode_rollout:
                        all_episode_rollout.append((name, transition)) 
                        num_eval_steps += 1
                        with self.write_lock:  
                            self.stored_transitions.append((name, transition, True)) 
                         


            with self.write_lock: 
                self.stored_ckpt_eval_transitions[ckpt] = all_episode_rollout  
                self.agent_ckpt_eval_summaries[ckpt] = all_agent_summaries
                self._finished_eval_checkpoint.append(ckpt)

            
            print(f'Checkpoint {ckpt} finished evaluating, all {len(self.stored_ckpt_eval_transitions[ckpt])} transitions and agent act summaries stored ') 
            if self._kill_signal.value and max(self._loaded_eval_checkpoint) == self.final_checkpoint_step:
                while len(self.stored_ckpt_eval_transitions.get(ckpt, [])) > 0:
                    time.sleep(1)
                    print('Process %s waiting for all evaled transitions to be logged' % name)
                print('Process %s shutting down after current ckpt is done evaluating' % name)
                env.shutdown()
                return 
        
    def kill(self):
        self._kill_signal.value = True

class EnvRunner(object):

    def __init__(self,
                 train_env: Env,
                 agent: Agent,
                 train_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer]],
                 num_train_envs: int,
                 num_eval_envs: int,
                 episodes: int,
                 episode_length: int,
                 eval_env: Union[Env, None] = None,
                 eval_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer], None] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 rollout_generator: RolloutGenerator = None,
                 weightsdir: str = None,
                 max_fails: int = 5,
                 device_list: Union[List[int], None] = None,
                 share_buffer_across_tasks: bool = True, 
                 share_buffer_across_vars: bool = True,  
                 task_var_to_replay_idx: dict = {},
                 eval_only: bool = False, 
                 iter_eval: bool = False, 
                 eval_episodes: int = 2,
                 log_freq: int = 100,
                 target_replay_ratio: float = 30.0,
                 final_checkpoint_step: int = 999,
                 dev_cfg: dict = None,
                 rew_cfg: dict = None,
                 ):
        self._train_env = train_env
        self._eval_env = eval_env if eval_env else deepcopy(train_env)
        self._agent = agent
        self._train_envs = num_train_envs
        self._eval_envs = num_eval_envs
        self._train_replay_buffer = train_replay_buffer if isinstance(train_replay_buffer, list) else [train_replay_buffer]
        self._timesteps = self._train_replay_buffer[0].timesteps
        if eval_replay_buffer is not None:
            eval_replay_buffer = eval_replay_buffer if isinstance(eval_replay_buffer, list) else [eval_replay_buffer]
        self._eval_replay_buffer = eval_replay_buffer
        self._episodes = episodes
        self._episode_length = episode_length
        self._stat_accumulator = stat_accumulator
        self._rollout_generator = (
            RolloutGenerator() if rollout_generator is None
            else rollout_generator)
        self._weightsdir = weightsdir
        self._max_fails = max_fails
        self._previous_loaded_weight_folder = ''
        self._p = None
        self._kill_signal = Value('b', 0)
        self._step_signal = Value('i', -1)
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        self._total_transitions = {'train_envs': 0, 'eval_envs': 0}
        self._total_episodes = {'train_envs': 0, 'eval_envs': 0} 
        self.target_replay_ratio = None  # Will get overridden later
        self.current_replay_ratio = Value('f', -1) 
        self.online_task_ids = Manager().list()
        self.buffer_add_counts = Manager().list()
        self.log_freq = log_freq
        self.target_replay_ratio = target_replay_ratio
        self._device_list = device_list  
        self._agent_summaries = []
        self._agent_ckpt_summaries = dict() 
        self.task_var_to_replay_idx = task_var_to_replay_idx

        self._all_task_var_ids = []
        self.replay_to_task_vars = collections.defaultdict(list)
        for task_id, var_dicts in task_var_to_replay_idx.items():
            self._all_task_var_ids.extend([(task_id, var_id) for var_id in var_dicts.keys() ])
            for var_id, replay_idx in var_dicts.items():
                self.replay_to_task_vars[replay_idx].append((task_id, var_id))
        logging.info(f'Counted a total of {len(self._all_task_var_ids)} variations')
        print('Replay index to task var mapping: ', self.replay_to_task_vars)

        
        self._eval_only = eval_only
        if eval_only:
            logging.info('Warning! Eval only, set number of training env to 0')
            self._train_envs = 0

        self._iter_eval = iter_eval
        self._eval_episodes = eval_episodes
        self.final_checkpoint_step = final_checkpoint_step
        self._dev_cfg = dev_cfg
        self._rew_cfg = rew_cfg
        
        self.clip_save_path = os.path.join(rew_cfg.data_path, rew_cfg.task_names[0], f'iteration{rew_cfg.save_itr}')
        if not rew_cfg.overwrite:
            assert not os.path.exists(self.clip_save_path), f'{self.clip_save_path} already exists!'

    @property   
    def device_list(self):
        # if self._device_list is None:
        #     return [i for i in range(torch.cuda.device_count())]
        # NOTE: if never given gpus at __init__, don't use gpus even if some are avaliable for agent training 
        return deepcopy(self._device_list)
    
    def summaries(self) -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop())
        for key, value in self._new_transitions.items():
            summaries.append(
                ScalarSummary('%s/new_transitions' % key, value))
        for key, value in self._total_transitions.items():
            summaries.append(
                ScalarSummary('%s/total_transitions' % key, value))
        for key, value in self._total_episodes.items():
            summaries.append(
                ScalarSummary('%s/total_episodes' % key, value))
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        summaries.extend(self._agent_summaries)
        
        return summaries

    def _update(self):
        # Move the stored transitions to the replay and accumulate statistics.
        new_transitions = collections.defaultdict(int)
        
        with self._internal_env_runner.write_lock:
            # logging.info('EnvRunner calling internal runner write lock')
            self._agent_summaries = list(
                self._internal_env_runner.agent_summaries)
            if self._step_signal.value % self.log_freq == 0 and self._step_signal.value > 0:
                self._internal_env_runner.agent_summaries[:] = []
            
            
            for name, transition, eval in self._internal_env_runner.stored_transitions:
                add_to_buffer = (not eval) or self._eval_replay_buffer is not None
                if self._train_envs == 0:
                    add_to_buffer = True # for PEARL agent, need to update buffer with eval transitions!
                if add_to_buffer:
                    kwargs = dict(transition.observation)
                    kwargs.update(transition.info)
                    # assert self._buffer_key in transition.info.keys(), \
                    #     f'Need to look for **{self._buffer_key}** in replay transition to know which buffer to add it to'
                    # replay_index = self.task_var_to_replay_idx.get(
                    #     transition.info[self._buffer_key], 0)
                    task_id = transition.info.get(TASK_ID, 0)
                    var_id = transition.info.get(VAR_ID, 0) 
                    replay_index = self.task_var_to_replay_idx[task_id][var_id] 
                    rb = self._train_replay_buffer[replay_index]
                    rb.add(
                        np.array(transition.action), transition.reward,
                        transition.terminal,
                        transition.timeout, **kwargs)
                    if transition.terminal:
                        rb.add_final(
                            **transition.final_observation)
                new_transitions[name] += 1
                self._new_transitions[
                    'eval_envs' if eval else 'train_envs'] += 1
                self._total_transitions[
                    'eval_envs' if eval else 'train_envs'] += 1
                
                if transition.terminal:
                    self._total_episodes['eval_envs' if eval else 'train_envs'] += 1

                if self._stat_accumulator is not None:
                    self._stat_accumulator.step(transition, eval)
            self._internal_env_runner.stored_transitions[:] = []  # Clear list
            # logging.info('Finished EnvRunner calling internal runner write lock')
            if len(self._internal_env_runner.clip_episodes) > 0:  
                os.makedirs(f'{self.clip_save_path}/success', exist_ok=True)
                os.makedirs(f'{self.clip_save_path}/fail', exist_ok=True) 
                for episode in self._internal_env_runner.clip_episodes:
                    folder = 'success' if episode[-1].info.get('task_success', False) else 'fail'
                    eps_idx = len(glob(f'{self.clip_save_path}/{folder}/episode*')) 
                    if eps_idx <= 5000:
                        new_path = f'{self.clip_save_path}/{folder}/episode{eps_idx}'
                        if eps_idx % 50 == 0:
                            print('Saving episode:', new_path)
                        os.makedirs(new_path, exist_ok=False)
                        for i, transition in enumerate(episode):
                            with open(f'{new_path}/{i}.pkl', 'wb') as f:
                                pickle.dump(transition, f)
                self._internal_env_runner.clip_episodes[:] = []
            
            for ckpt_step, all_transitions in self._internal_env_runner.stored_ckpt_eval_transitions.items():
                if ckpt_step in self._internal_env_runner._finished_eval_checkpoint:  
                    for name, transition in all_transitions:
                        self._new_transitions['eval_envs'] += 1
                        self._total_transitions['eval_envs'] += 1
                        if transition.terminal:
                            self._total_episodes['eval_envs'] += 1 
                    
                    self._agent_ckpt_summaries[ckpt_step] = self._internal_env_runner.agent_ckpt_eval_summaries.pop(ckpt_step, [])
                    if self._stat_accumulator is not None:
                        self._stat_accumulator.step_all_transitions_from_ckpt(all_transitions, ckpt_step)
                    self._internal_env_runner.stored_ckpt_eval_transitions.pop(ckpt_step, []) # Clear 
                    

                    logging.debug('Done poping ckpt {} eval transitions to accumulator, main EnvRunner stored {} agent summaries, remaining ckpts: '.format(
                        ckpt_step, 
                        len(self._agent_ckpt_summaries.get(ckpt_step, []))), 
                        self._internal_env_runner.stored_ckpt_eval_transitions.keys() 
                        )

            self.buffer_add_counts[:] = [int(r.add_count) for r in self._train_replay_buffer]
            demo_cursor = self._train_replay_buffer[0]._demo_cursor
            if demo_cursor > 0: # i.e. only on-line samples can be used for context
                self.buffer_add_counts[:] = [int(r.add_count - r._demo_cursor) for r in self._train_replay_buffer]
            self._internal_env_runner.online_buff_id.value = -1 
            # if self._train_replay_buffer[0].batch_size > min(self.buffer_add_counts): 
            #     buff_id = np.argmin(self.buffer_add_counts) 
            #     print('Setting buffer id to prioritize low-count buffers', buff_id)
            #     self._internal_env_runner.online_buff_id.value = buff_id
             
        return new_transitions
 
    def try_log_ckpt_eval(self):
        """Attempts to log the earliest avaliable ckpt that finished eval"""
        ckpt, summs = self._stat_accumulator.pop_ckpt_eval() 
        if ckpt > -1:
            assert ckpt in self._agent_ckpt_summaries.keys(), 'Checkpoint has env transitions all stepped in accumulator but no agent summaries found' 
            summs += self._agent_ckpt_summaries.pop(ckpt, [])
        return ckpt, summs 

    def _run(self, save_load_lock):
        self._internal_env_runner = _EnvRunner(
            train_env=self._train_env, eval_env=self._eval_env, agent=self._agent, timesteps=self._timesteps, train_envs=self._train_envs,
            eval_envs=self._eval_envs, episodes=self._episodes, episode_length=self._episode_length, kill_signal=self._kill_signal,
            step_signal=self._step_signal, rollout_generator=self._rollout_generator, save_load_lock=save_load_lock,
            current_replay_ratio=self.current_replay_ratio, 
            target_replay_ratio=self.target_replay_ratio, 
            online_task_ids=self.online_task_ids,
            weightsdir=self._weightsdir, 
            device_list=(self.device_list if len(self.device_list) >= 1 else None),
            all_task_var_ids=self._all_task_var_ids,
            eval_episodes=self._eval_episodes,
            final_checkpoint_step=self.final_checkpoint_step, 
            )
        #training_envs = self._internal_env_runner.spin_up_envs('train_env', self._train_envs, False)
        #eval_envs = self._internal_env_runner.spin_up_envs('eval_env', self._eval_envs, True)
        #envs = training_envs + eval_envs
        envs = self._internal_env_runner.spinup_train_and_eval(self._train_envs, self._eval_envs, 'env', iter_eval=self._iter_eval)
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
                        p = self._internal_env_runner.restart_process(p.name)
                        envs.append(p)

            if not self._kill_signal.value or len(self._internal_env_runner.stored_transitions) > 0 or \
                 len(self._internal_env_runner.stored_ckpt_eval_transitions) > 0:
                new_transitions = self._update()
                for p in envs:
                    if new_transitions[p.name] == 0:
                        no_transitions[p.name] += 1
                    else:
                        no_transitions[p.name] = 0
                    if no_transitions[p.name] > WAIT_TIME:  # 5min
                        if self.current_replay_ratio.value - 1 > self.target_replay_ratio:
                            # only hangs if it Should be running, otherwise just let it sleep? 
                            logging.warning("Env %s hangs, so restarting" % p.name)
                            print('process id:', p.pid)
                            print('process is alive?', p.is_alive())
                            print('replay&target ratios:', self.current_replay_ratio.value, self.target_replay_ratio)
                            envs.remove(p)
                            os.kill(p.pid, signal.SIGTERM)
                            torch.cuda.empty_cache()
                            p = self._internal_env_runner.restart_process(p.name)
                            envs.append(p)
                            no_transitions[p.name] = 0

            if len(envs) == 0:
                break
            time.sleep(1)

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
