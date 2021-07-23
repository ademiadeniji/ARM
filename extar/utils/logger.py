import csv
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from yarr.agents.agent import Summary, ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary
 
from multiprocessing import Lock
from typing import List
from yarr.utils.transition import ReplayTransition
from yarr.utils.stat_accumulator import StatAccumulator, _SimpleAccumulator
import wandb 


class WandbLogWriter(object):
    """Only do wandb + csv for backup """
    def __init__(self, logdir: str, csv_logging: bool=True): 
        self._csv_logging = csv_logging
        os.makedirs(logdir, exist_ok=True) 
 
        self._prev_row_data = self._row_data = OrderedDict()
        self._csv_file = os.path.join(logdir, 'data.csv')
        self._field_names = None

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

            self._prev_row_data = self._row_data
            self._row_data = OrderedDict()

    def close(self):
        # if self._tensorboard_logging:
        #     self._tf_writer.close()
        return 


class MultiTaskAccumulator(StatAccumulator):
    
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
        for acc in self._train_accs.values():
            combined.extend(acc.pop())
        for acc in self._eval_accs.values():
            combined.extend(acc.pop())
        return combined

    def peak(self): # -> List[Summary]:
        combined = self._train_accs_mean.peak()
        for acc in self._train_accs.values():
            combined.extend(acc.peak())
        for acc in self._eval_accs.values():
            combined.extend(acc.peak())
        return combined

    def reset(self): # -> None:
        self._train_accs_mean.reset()
        for acc in self._train_accs.values():
            acc.reset() 
        for acc in self._eval_accs.values():
            acc.reset() 