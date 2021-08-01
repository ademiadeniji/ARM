import os
import time
import copy
import wandb 
import torch
import torch.nn as nn
import logging 
import shutil
import signal
import sys
import threading
import time
from multiprocessing import Lock 

import numpy as np
import psutil
import torch

from copy import deepcopy 
from collections import OrderedDict, defaultdict
from multiprocessing import Process, Manager, Value
from typing import Any, List, Union, Optional 
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.envs.env import Env  

from arm.custom_rlbench_env import CustomRLBenchEnv, MultiTaskRLBenchEnv
from yarr.utils.stat_accumulator import StatAccumulator
 
from extar.runners.multi_env_runner import MultiTaskEnvRunner
from extar.utils.logger import WandbLogWriter, MultiTaskAccumulator
from yarr.agents.agent import Summary, ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary
import wandb 

NUM_WEIGHTS_TO_KEEP = 20

from abc import abstractmethod, ABC
from typing import Union, List

from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers import WrappedReplayBuffer
 

class MultiTaskEvalRunner(object):
    """Mainly building off the Trainer but just to evaluation things here"""
    def __init__(self,
                agent:          Agent,
                env_runner:     MultiTaskEnvRunner,
                replays,
                train_device:   torch.device, 
                device_list:    Union[List[int], None] = None,
                replay_buffer_sample_rates: List[float] = None,
                stat_accumulator: Union[MultiTaskAccumulator, None] = None,
                iterations: int = int(1e6),
                logdir: str = '/tmp/yarr/logs',
                log_freq: int = 10,
                transitions_before_train: int = 1000,
                weightsdir: str = '/tmp/yarr/weights',
                save_freq: int = 100,
                replay_ratio: Optional[float] = None,
                csv_logging: bool = True 
                ):
        self._agent = agent
        self._env_runner = env_runner
        self._stat_accumulator = stat_accumulator
        self._iterations = iterations
        self._logdir = logdir
        self._log_freq = log_freq
        self._transitions_before_train = int(transitions_before_train / len(replays))
        self._weightsdir = weightsdir
        self._save_freq = save_freq
       
        env_runner.log_freq = log_freq
        env_runner.target_replay_ratio = replay_ratio
        self._replays = replays
        self._task_names = sorted(replays.keys())
        self._replay_list = [self._replays.get(name) for name in self._task_names]
        
         
        self._replay_buffer_sample_rates = replay_buffer_sample_rates
        if replay_buffer_sample_rates == [1.0] and len(replays) > 1:
            self._replay_buffer_sample_rates = [1.0/len(replays) for r in replays.keys()]
            print('Setting same sampling rates for all tasks to:',  self._replay_buffer_sample_rates)
        else:
            if sum(self._replay_buffer_sample_rates) != 1:
                raise ValueError('Sum of sampling rates should be 1.')
        if len(self._replay_buffer_sample_rates) != len(replays):
            raise ValueError(
                'Numbers of replay buffers differs from sampling rates.')
        
        self._train_device = train_device
        self._device_list = device_list 
        self._csv_logging = csv_logging

        if replay_ratio is not None and replay_ratio < 0:
            raise ValueError("max_replay_ratio must be positive.")
        self._target_replay_ratio = replay_ratio

        self._writer = None
        if logdir is None:
            logging.info("Warning! 'logdir' is None. No logging will take place.")
        else:
            self._writer = WandbLogWriter(self._logdir, csv_logging)
        if weightsdir is None:
            logging.info("Warning! 'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)

        self.accumulate_times = {'sample': 0, 'agent_step': 0, 'env_step': 0}
    
    @property   
    def device_list(self):
        if self._device_list is None:
            return [i for i in range(torch.cuda.device_count())]
        return deepcopy(self._device_list)

    def _save_model(self, i):
        """Copied from PyTorchTrainRunner """
        print('Debugging: saving model at step', i)
        with self._save_load_lock:
            d = os.path.join(self._weightsdir, str(i))
            os.makedirs(d, exist_ok=True)
            if isinstance(self._agent, nn.DataParallel):
                self._agent.module.save_weights(d)
            else:
                self._agent.save_weights(d)
            # Remove oldest save
            prev_dir = os.path.join(self._weightsdir, str(
                i - self._save_freq * NUM_WEIGHTS_TO_KEEP))
            if os.path.exists(prev_dir):
                shutil.rmtree(prev_dir)

    def _step(self, i, sampled_batch):
        update_dict = self._agent.update(i, sampled_batch)
        acc_bs = 0
        for wb in self._replay_list:
            bs = wb.replay_buffer.batch_size
            if 'priority' in update_dict:
                wb.replay_buffer.set_priority(
                    sampled_batch['indices'][acc_bs:acc_bs+bs].cpu().detach().numpy(),
                    update_dict['priority'][acc_bs:acc_bs+bs])
            acc_bs += bs
    
    def _signal_handler(self, sig, frame):
        if threading.current_thread().name != 'MainThread':
            return
        logging.info('SIGINT captured. Shutting down.'
                     'This may take a few seconds.')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._replay_list]
        sys.exit(0)

    def _get_add_counts(self):
        return np.array( [r.replay_buffer.add_count for r in self._replay_list] )

    def _get_sum_add_counts(self):
        return sum([ r.replay_buffer.add_count for r in self._replay_list] )
    
    def generate_video(self, vid_sequence):
        eps_vid, attn_vid = [], []
        for pair in vid_sequence:
            agent_attn, env_img = pair 
            for attn in agent_attn:
                print(type(attn), attn.value.shape)
            print('env recorded img shape:', env_img.shape)
            eps_vid.append(env_img)
             
            attn_vid.append(agent_attn[-1].value)

        return np.stack(eps_vid), np.stack(attn_vid)
    
    def start(self, load_dir=None):
    
        signal.signal(signal.SIGINT, self._signal_handler)

        self._save_load_lock = Lock()

        # Kick off the environments
        self._env_runner.start(self._save_load_lock)

        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device)
        assert load_dir is not None
        print('Loading weights')
        self._agent.load_weights(load_dir)
        # if len(self.device_list) > 1:
        #     self._agent = nn.DataParallel(self._agent)

        assert self._weightsdir is not None, 'Must save that loaded checkpoint to a new dir'
        self._save_model(0)  # Save weights so workers can load.

        #logging.info('Waiting for %d samples before training. After demos, currently have %s.' %
        #        (self._transitions_before_train, str(self._get_add_counts())))
        while (np.any(self._get_add_counts() < self._transitions_before_train)):
            time.sleep(1)
            if self._env_runner._videos is not None:
                videos = self._env_runner._videos
                eps_vid, attn_vid = self.generate_video(videos)
                wandb.log({'video_log_step': 0, 'episode_video': eps_vid, 'attention_video': attn_vid})
                raise ValueError
        #    logging.info( 'Waiting for %d samples before training. Currently have %s.' %
                # (self._transitions_before_train, str(self._get_add_counts())))
        # if (np.all(self._get_add_counts() > self._transitions_before_train)):
                    
        
        logging.info('Finished adding all %d samples before training. Currently have %s.' %
                (self._transitions_before_train, str(self._get_add_counts())))

        # datasets = [r.dataset() for r in self._replay_list]
        # data_iter = [iter(d) for d in datasets]
        # get_idx_to_task = {
            #int(i * self._replay_list[name].replay_buffer.batch_size): name for i, name in enumerate(self._task_names) }

        init_replay_size = self._get_sum_add_counts().astype(float)
        batch_size = sum([r.replay_buffer.batch_size for r in self._replay_list])
        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        for i in range(self._iterations):
            
            if i > 0:
                runner_time =  time.time() - self._env_runner.last_step_time
                self.accumulate_times['env_step'] += runner_time 

            self._env_runner.set_step(i) 
            # params = [qfunc._q.state_dict() for qfunc in self._agent._pose_agent._qattention_agents]
            # self._env_runner.recieve_agent(copy.deepcopy(params), i)
            
            
            log_iteration = i % self._log_freq == 0  
            if log_iteration:
                process.cpu_percent(interval=None)
 

            t = time.time()
            # sampled_batch = [next(di) for di in data_iter]
            #if len(sampled_batch) > 1:
             #   result = {}
            #    for key in sampled_batch[0]:
            #        result[key] = torch.cat([d[key] for d in sampled_batch], 0)
            #    sampled_batch = result
            #else:
            #    sampled_batch = sampled_batch[0]

            #sample_time = time.time() - t
            #self.accumulate_times['sample'] += sample_time
            #batch = {k: v.to(self._train_device) for k, v in sampled_batch.items()}
            #t = time.time()
            #self._step(i, batch)
            #step_time = time.time() - t
            #self.accumulate_times['agent_step'] += 0 #step_time 
            
            if log_iteration and self._writer is not None:
                #replay_ratio = get_replay_ratio()
                # logging.info('Step %d. ReplaySample time: %.2f. EnvStep time: %.2f. AgentUpdate time: %.2f. Replay ratio: %.3f' % (
                #              i, 
                #              self.accumulate_times['sample'], 
                #              self.accumulate_times['env_step'],
                #              self.accumulate_times['agent_step'], 
                #              replay_ratio))
                #agent_summaries = self._agent.update_summaries()
                env_summaries = self._env_runner.summaries()
                self._writer.add_summaries(i, env_summaries) #agent_summaries + env_summaries)
 
                #scalar_dict = {
                    #'replay/replay_ratio':              replay_ratio,
                    #'replay/update_to_insert_ratio':    float(i) / float(self._get_sum_add_counts() - init_replay_size + 1e-6),
                #    'monitoring/sample_time_per_item':  sample_time / batch_size,
                #    'monitoring/train_time_per_item':   step_time / batch_size,
                #   'monitoring/memory_gb':             process.memory_info().rss * 1e-9,
                #    'monitoring/cpu_percent':           process.cpu_percent(interval=None) / num_cpu,
                #    'monitoring/accum_agent_update_time': self.accumulate_times['agent_step'],
                #    'monitoring/accum_env_step_time': self.accumulate_times['env_step'],
                #   'monitoring/accum_sample_time': self.accumulate_times['sample'],

                #}
                #self._writer.add_scalar_dict(i, scalar_dict)

            self._writer.end_iteration()

            # if i % self._save_freq == 0 and self._weightsdir is not None:
            #    self._save_model(i)

        if self._writer is not None:
            self._writer.close()

        logging.info('Stopping envs ...')
        self._env_runner.stop()
         