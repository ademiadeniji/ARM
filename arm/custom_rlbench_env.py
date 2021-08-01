import numpy as np
from typing import Type, List
from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy
from rlbench import ObservationConfig, CameraConfig
from rlbench.action_modes import ActionMode
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.task_environment import InvalidActionError

from yarr.agents.agent import ActResult, VideoSummary
from yarr.envs.rlbench_env import RLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

# New(Mandi):
from rlbench.backend.utils import task_file_to_task_class 
from hydra.utils import instantiate
from collections import OrderedDict
from copy import deepcopy
import random 
RECORD_EVERY = 20


class CustomRLBenchEnv(RLBenchEnv):

    def __init__(self,
                 task_class: Type[Task],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,):
        super(CustomRLBenchEnv, self).__init__(
            task_class, observation_config, action_mode, dataset_root,
            channels_last, headless=headless)
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._i = 0

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 2,)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)
        obs.gripper_matrix = grip_mat
        obs.gripper_pose = grip_pose
        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._previous_obs_dict = super(CustomRLBenchEnv, self).reset()
        self._record_current_episode = (
                self.eval and self._episode_index % RECORD_EVERY == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        self._i = 0
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.
        try:
            obs, reward, terminal = self._task.step(action)
            if terminal:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = -1.0

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(VideoSummary('episode_rollout', vid, fps=30))
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i):
        d, = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)
        self._task.reset_to_demo(d)


class MultiTaskRLBenchEnv(CustomRLBenchEnv):
    def __init__(self,
                 train_tasks: List[str],
                 eval_tasks: List[str], 
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 state_includes_remaining_time: bool = True,
                 include_previous_action: bool = False,
                 sample_method: str = 'uniform'
                 ):
        self.train_tasks = train_tasks
        self.train_task_classes = {
            name: task_file_to_task_class(name) for name in train_tasks }

        self.eval_tasks = eval_tasks 
        self.unique_tasks = {
            name: task_file_to_task_class(name) for name in \
                set(train_tasks + eval_tasks)} # for easier env runner 
        super(MultiTaskRLBenchEnv, self).__init__(
            None, # NOTE: setting task class to None at init, only sample task during self.launch() and self.reset()
            observation_config, 
            action_mode, 
            episode_length,
            dataset_root,
            channels_last, 
            reward_scale,
            headless,
            state_includes_remaining_time,
            include_previous_action)
        
        self.n_train_tasks, self.n_eval_tasks = len(train_tasks), len(eval_tasks)
        self.n_unique_tasks = len( set(train_tasks + eval_tasks) )
        self.sample_method = sample_method # TODO(Mandi): add non-uniform sampling

    def reset_task(self):
        """Sample a task name first, then reset """
        if self.sample_method == 'uniform':
            task_names = self.eval_tasks if self.eval else self.train_tasks 
            self._task_name = random.sample(task_names, k=1)[0]
            task_class = task_file_to_task_class(self._task_name)
        else:
            raise NotImplementedError
        #self._task = self._rlbench_env.get_task(task_class)
        self._task_class = task_class 

    def step(self, act_result: ActResult): # -> Transition:
        """Also return task name"""
        transition = super(MultiTaskRLBenchEnv, self).step(act_result)
        transition.info['task_name'] = self._task_name 
        return transition 

    def launch(self):
        self.reset_task()
        self._rlbench_env.launch()
        self._task = self._rlbench_env.get_task(self._task_class)
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self): # -> dict:
        """Also re-sample a potentionally new task"""
        self._record_current_episode = (
                self.eval and self._episode_index % RECORD_EVERY == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        self._i = 0
        self._prev_action = None
        self.reset_task()
        descriptions, obs = self._task.reset()
        self._previous_obs = obs
        self._previous_obs_dict = self.extract_obs(obs)
        return self._previous_obs_dict
