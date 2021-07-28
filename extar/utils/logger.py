import csv
import logging
import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from yarr.agents.agent import Summary, ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary
 
from multiprocessing import Lock
from typing import List
from yarr.utils.transition import ReplayTransition

import wandb 


class WandbLogWriter(object):
    """Only do wandb + csv for backup """
    def __init__(self, logdir: str, csv_logging: bool=True): 
        self._csv_logging = csv_logging
        os.makedirs(logdir, exist_ok=True) 
 
        self._prev_row_data = self._row_data = OrderedDict()
        self._csv_file = os.path.join(logdir, 'data.csv') 
        self._field_names = None
        self._curr_image = None

    def add_scalar(self, i, name, value):  
 
        if len(self._row_data) == 0:
            self._row_data['step'] = i
        self._row_data[name] = value.item() if isinstance(
            value, torch.Tensor) else value

    def add_scalar_dict(self, i, scalar_dict):
        if len(self._row_data) == 0:
            self._row_data['step'] = i
        for name, value in scalar_dict.items(): 
            self._row_data[name] = value.item() if isinstance(value, torch.Tensor) else value


    def add_summaries(self, i, summaries):
        for summary in summaries:
            try:
                if isinstance(summary, ScalarSummary):
                    self.add_scalar(i, summary.name, summary.value)
                elif isinstance(summary, ImageSummary):  # Only grab first item in batch
                    v = summary.value if summary.value.ndim == 3 else summary.value[0]
                    self._curr_image = (i, v)
                # elif self._tensorboard_logging: TODO(Mandi) move to wandb
                #     if isinstance(summary, HistogramSummary):
                #         self._tf_writer.add_histogram(
                #             summary.name, summary.value, i)
                #     elif isinstance(summary, ImageSummary):
                #         # Only grab first item in batch
                #         v = (summary.value if summary.value.ndim == 3 else
                #              summary.value[0])
                #         self._tf_writer.add_image(summary.name, v, i)
                #     elif isinstance(summary, VideoSummary):
                #         # Only grab first item in batch
                #         v = (summary.value if summary.value.ndim == 5 else
                #              np.array([summary.value]))
                #         self._tf_writer.add_video(
                #             summary.name, v, i, fps=summary.fps)
            except Exception as e:
                logging.error('Error on summary: %s' % summary.name)
                raise e

    def end_iteration(self):
        if len(self._row_data) > 0:
            with open(self._csv_file, mode='a+') as csv_f:
                names = self._row_data.keys() #or self._field_names  
                #print(self._field_names, self._row_data.keys())
                #print(np.array_equal(self._field_names, self._row_data.keys()))
                writer = csv.DictWriter(csv_f, fieldnames=names)
                if self._field_names is None:
                    writer.writeheader()
                else:
                    if not np.array_equal(self._field_names, self._row_data.keys()):
                        # Special case when we are logging faster than new
                        # summaries are coming in.
                        missing_keys = list(set(self._field_names) - set(
                            self._row_data.keys()))
                        for mk in missing_keys:
                            self._row_data[mk] = self._prev_row_data[mk]
                self._field_names = names
                #hack
                 
                writer.writerow(self._row_data)

            wandb.log(self._row_data)
            if self._curr_image is not None:
                (itr, img) = self._curr_image
                wandb.log({'image-itr': itr, 'image': wandb.Image(img)})

            self._prev_row_data = self._row_data
            self._row_data = OrderedDict()

    def close(self):
        # if self._tensorboard_logging:
        #     self._tf_writer.close()
        return 

from multiprocessing import Lock
from typing import List

import numpy as np
from yarr.agents.agent import Summary, ScalarSummary
from yarr.utils.transition import ReplayTransition


class Metric(object):

    def __init__(self):
        self._previous = []
        self._current = 0

    def update(self, value):
        self._current += value

    def next(self):
        self._previous.append(self._current)
        self._current = 0

    def reset(self):
        self._previous.clear()

    def min(self):
        return np.min(self._previous)

    def max(self):
        return np.max(self._previous)

    def mean(self):
        return np.mean(self._previous)

    def median(self):
        return np.median(self._previous)

    def std(self):
        return np.std(self._previous)

    def __len__(self):
        return len(self._previous)

    def __getitem__(self, i):
        return self._previous[i]


class _SimpleAccumulator(object):

    def __init__(self, prefix, eval_video_fps: int = 30,
                 mean_only: bool = True):
        self._prefix = prefix
        self._eval_video_fps = eval_video_fps
        self._mean_only = mean_only
        self._lock = Lock()
        self._episode_returns = Metric()
        self._episode_lengths = Metric()
        self._summaries = []
        self._transitions = 0

    def _reset_data(self):
        with self._lock:
            self._episode_returns.reset()
            self._episode_lengths.reset()
            self._summaries.clear()

    def step(self, transition: ReplayTransition, eval: bool):
        with self._lock:
            self._transitions += 1
            self._episode_returns.update(transition.reward)
            self._episode_lengths.update(1)
            if transition.terminal:
                self._episode_returns.next()
                self._episode_lengths.next()
            self._summaries.extend(list(transition.summaries))

    def _get(self) -> List[Summary]:
        sums = []

        if self._mean_only:
            stat_keys = ["mean"]
        else:
            stat_keys = ["min", "max", "mean", "median", "std"]
        names = ["return", "length"]
        metrics = [self._episode_returns, self._episode_lengths]
        for name, metric in zip(names, metrics):
            for stat_key in stat_keys:
                if self._mean_only:
                    assert stat_key == "mean"
                    sum_name = '%s/%s' % (self._prefix, name)
                else:
                    sum_name = '%s/%s/%s' % (self._prefix, name, stat_key)
                sums.append(
                    ScalarSummary(sum_name, getattr(metric, stat_key)()))
        sums.append(ScalarSummary(
            '%s/total_transitions' % self._prefix, self._transitions))
        sums.extend(self._summaries)
        return sums

    def pop(self) -> List[Summary]:
        data = []
        if len(self._episode_returns) > 1:
            data = self._get()
            self._reset_data()
        return data

    def peak(self) -> List[Summary]:
        return self._get()
    
    def reset(self):
        self._transitions = 0
        self._reset_data()


class MultiTaskAccumulator(object):
    
    def __init__(self, train_tasks, eval_tasks,
                 eval_video_fps: int = 30, 
                 mean_only: bool = True):
        self._train_accs = {
            task_name: _SimpleAccumulator('train_%s/envs' % (task_name), eval_video_fps, mean_only=mean_only)
            for task_name in train_tasks
            }
        self._eval_accs = {
            eval_name: _SimpleAccumulator('eval_%s/envs' % (eval_name), eval_video_fps, mean_only=mean_only)
            for eval_name in eval_tasks
            }
            
        self._train_accs_mean = _SimpleAccumulator(
            'train_summary/envs', eval_video_fps, mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        task_name = transition.info.get('task_name', None)
        assert task_name, f'Task {task_name} does not have stat accumulator!'
        if eval:
            self._eval_accs[task_name].step(transition, eval)
        else:
            self._train_accs[task_name].step(transition, eval)
            self._train_accs_mean.step(transition, eval)

    def pop(self): # -> List[Summary]:
        combined = self._train_accs_mean.pop()
        popped_train, popped_eval = defaultdict(list), defaultdict(list)
        for task_name, acc in self._train_accs.items():
            #popped_train[task_name].extend(acc.pop())
            combined.extend(acc.pop())
        for task_name, acc in self._eval_accs.items():
            #popped_eval[task_name].extend(acc.pop())
            combined.extend(acc.pop())
        return combined 

    # def peak(self): # -> List[Summary]:
    #     combined = self._train_accs_mean.peak()
    #     peaked_train, peaked_eval = defaultdict(list), defaultdict(list)
    #     for task_name, acc in self._train_accs.items():
    #         peaked_train[task_name].extend(acc.peak())
    #     for acc in self._eval_accs.items():
    #         peaked_eval.extend(acc.peak())
    #     return combined, peaked_train, peaked_eval 

    def reset(self): # -> None:
        self._train_accs_mean.reset()
        for acc in self._train_accs.values():
            acc.reset() 
        for acc in self._eval_accs.values():
            acc.reset() 