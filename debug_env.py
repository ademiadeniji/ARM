import os 
import torch 
from omegaconf import DictConfig, OmegaConf, ListConfig
from arm.custom_rlbench_env_multitask import CustomMultiTaskRLBenchEnv
from launch_multitask import _create_obs_config 
from launch_context import ACTION_MODE
from rlbench.backend.utils import task_file_to_task_class
from yarr.agents.agent import Agent, ActResult
import numpy as np
# from pynvml import *
# nvmlInit()
# watch -n30 nvidia-smi  !!! 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["QT_LOGGING_RULES" ] = '*.debug=false;qt.qpa.*=false'
os.environ["XDG_RUNTIME_DIR"] = '/tmp/runtime-mandi'

display_device = os.environ["DISPLAY"].split('.')[-1]
print('Using device {} for display'.format(display_device))
os.environ["CUDA_VISIBLE_DEVICES"] = str(display_device)

tasks_cfg = OmegaConf.load('/home/mandi/ARM/conf/tasks/MT10_1cam.yaml')
tasks = sorted([t for t in tasks_cfg.all_tasks])
task_classes = [task_file_to_task_class(t) for t in tasks]
obs_config = _create_obs_config(['front'],[128, 128]) 

# h = nvmlDeviceGetHandleByIndex(0)


env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes, task_names=tasks, observation_config=obs_config,
        action_mode=ACTION_MODE, dataset_root='/shared/mandi/all_rlbench_data',
        episode_length=10, headless=True, 
        use_variations=[] # all variations
        )
print('launching task env')

env.eval = True 
env.launch()
act = np.array([
    0.49687502, -0.328125, 0.75937498, -0.74119243, \
    0.15134651, -0.56754425, -0.32499469, 1. ]
    ).reshape((8,))
env._record_current_episode = True
for i in range(1000):
    obs = env.reset(swap_task=True ) 
    trans = env.step(
        ActResult(
            action=act,
            observation_elements={}, 
            info={})
            )
    if i % 50 == 0:
        # info = nvmlDeviceGetMemoryInfo(h)
        # print('GPU memory usage: {}'.format(info.used))
        
        print('Switching to task: ',env._active_task_id)
