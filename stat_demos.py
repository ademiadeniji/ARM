from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes import ActionMode
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

import os
import pickle as pkl
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np
from glob import glob
from os.path import join 
import numpy as np 

DATA_DIR = '/home/mandi/all_rlbench_data' # /shared/mandi/all_rlbench_data 
DEMOS_PER_VARIATIONS = 20 

def main():
    all_names = sorted([ s.split('/')[-1] for s in  glob(join(DATA_DIR, '*')) ])
    print(f'Found {len(all_names)} task names, after sorting, first/last tasks are: {all_names[0]}, {all_names[-1]}')
    infos = dict()
    all_lengths = []
    all_vars = []
    need_demos = dict()
    for name in all_names:
        task_info = dict()
        variations = glob( join(DATA_DIR, name, 'variation*') )
        if len(variations) == 0:
            print(f'Warning! No variation found for {name}, variations')
        all_eps = glob( join(DATA_DIR, name, '*', 'episodes/*') ) 
        if len(all_eps) < len(variations) * DEMOS_PER_VARIATIONS:
            need_demos[name] = (len(all_eps), len(variations))
        #print(all_eps) 
        lengths = []
        for eps in all_eps:
            all_pngs = glob( join(eps, 'front_rgb/*') )
            lengths.append(len(all_pngs))
        if len(lengths) == 0:
            lengths = [-1]
        task_info = {
        'num_vars': len(variations),
        'num_episodes': len(all_eps),
        'episode_min_max_mean': ( np.min(lengths), np.max(lengths), np.mean(lengths) )
        }
        if len(lengths) == 1:
            print(f'Warning! No episode found for  Task {name}', all_eps)
        stats = task_info['episode_min_max_mean']
        print(f'Task {name}: across {len(variations)} variations and {len(all_eps)} episodes, length min: {stats[0]:0.2f}, max: {stats[1]:0.2f}, avg {stats[2]:0.2f}') 
        all_lengths.extend(lengths)
        all_vars.extend(variations)
        infos[name] = task_info 
    
    min_all, max_all, avg_all = np.min(all_lengths), np.max(all_lengths), np.mean(all_lengths)
    print('---'*10)
    print(f'Across all tasks, {len(all_vars)} variations,  min, max, mean episode lengths are {min_all}, {max_all}, {avg_all}') 
    for k, pair in need_demos.items():
        print(f'Task {k} need a total of {DEMOS_PER_VARIATIONS * pair[1]} demos for {pair[1]} variations, currently has {pair[0]}')

if __name__ == '__main__':
    main()

