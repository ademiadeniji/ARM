from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench import CameraConfig, ObservationConfig, ArmActionMode
from rlbench.action_modes import ActionMode, GripperActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import MT30_V1
from rlbench.tasks import PickUpCup, PickAndLift
import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        arm = np.array([np.nan for _ in range(self.action_size - 1)])
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

camera_names = ['front']
obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
# action_mode = ActionMode(
#         ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME,
#         GripperActionMode.OPEN_AMOUNT)
env = Environment(
    action_mode, obs_config=obs_config, headless=True) # need to set headless to True! 
print('Starting to launch')
env.launch()
print('Finish to launch')

agent = Agent(env.action_size)

train_tasks = [PickAndLift, PickUpCup] #  MT30_V1['train'] #

training_cycles_per_task = 30
training_steps_per_task = 10000
episode_length = 2

for _ in range(training_cycles_per_task):

    task_to_train = np.random.choice(train_tasks, 1)[0]
    task = env.get_task(task_to_train) 

    for i in range(training_steps_per_task):
        if i % episode_length == 0:
            _var = task.sample_variation()  # random variation
            print(f'Reset Episode to var number {_var}')
            descriptions, obs = task.reset()
            # print(descriptions)
        action = agent.act(obs)
        obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()