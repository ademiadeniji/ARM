from typing import Type, List

import numpy as np
from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy
from rlbench import ObservationConfig, CameraConfig
from rlbench.action_modes import ActionMode
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.task_environment import InvalidActionError

from yarr.agents.agent import ActResult, VideoSummary
from yarr.envs.rlbench_env import RLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

from collections import OrderedDict
from copy import deepcopy
from hydra.utils import instantiate
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
                 headless: bool = True,
                 state_includes_remaining_time: bool = True,
                 include_previous_action: bool = False):
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
        self._state_includes_remaining_time = state_includes_remaining_time
        self._include_previous_action = include_previous_action
        self._i = 0
        self._prev_action = None

    @property
    def observation_elements(self): # -> List[ObservationElement]:
        elements = []
        low_dim_state_len = 0
        if self._observation_config.joint_positions:
            low_dim_state_len += 7
        if self._observation_config.joint_forces:
            low_dim_state_len += 7
        if self._observation_config.gripper_open:
            low_dim_state_len += 1
        if self._observation_config.gripper_joint_positions:
            low_dim_state_len += 2
        if self._observation_config.gripper_touch_forces:
            low_dim_state_len += 6
        if self._observation_config.task_low_dim_state:
            raise NotImplementedError()

        if self._state_includes_remaining_time:
            low_dim_state_len += 1

        if self._include_previous_action:
            low_dim_state_len += 8

        if low_dim_state_len > 0:
            elements.append(ObservationElement(
                'low_dim_state', (low_dim_state_len, ), np.float32))
        elements.extend(self._get_cam_observation_elements(
            self._observation_config.left_shoulder_camera, 'left_shoulder'))
        elements.extend(self._get_cam_observation_elements(
            self._observation_config.right_shoulder_camera, 'right_shoulder'))
        elements.extend(self._get_cam_observation_elements(
            self._observation_config.front_camera, 'front'))
        elements.extend(self._get_cam_observation_elements(
            self._observation_config.wrist_camera, 'wrist'))
        elements.extend(self._get_cam_observation_elements(
            self._observation_config.overhead_camera, 'overhead'))
        self.low_dim_state_len = low_dim_state_len

        return elements

    def set_task(self, task_class: Type[Task]):
        self._task = self._rlbench_env.get_task(task_class)

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None

        # Turn gripper quaternion to be positive w
        if obs.gripper_pose is not None and obs.gripper_pose[-1] < 0:
            obs.gripper_pose[3:] *= -1.0

        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        if self._state_includes_remaining_time:
            tt = 1. - ((self._i if t is None else t) / self._episode_length)
            obs_dict['low_dim_state'] = np.concatenate([obs_dict['low_dim_state'], [tt]]).astype(np.float32)

        if self._include_previous_action:
            pa = self._prev_action if prev_action is None else prev_action
            pa = np.zeros((8,)) if pa is None else pa
            obs_dict['low_dim_state'] = np.concatenate([obs_dict['low_dim_state'], pa]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        obs.gripper_pose = grip_pose
        for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
            obs_dict[k] = v.astype(np.float32)

        for config, name in [
            (self._observation_config.left_shoulder_camera, 'left_shoulder'),
            (self._observation_config.right_shoulder_camera, 'right_shoulder'),
            (self._observation_config.front_camera, 'front'),
            (self._observation_config.wrist_camera, 'wrist'),
            (self._observation_config.overhead_camera, 'overhead')]:
            if config.point_cloud:
                obs_dict['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
                obs_dict['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]

        return obs_dict

    def _get_cam_observation_elements(self, camera: CameraConfig, prefix: str):
        elements = []
        img_s = list(camera.image_size)
        shape = img_s + [3] if self._channels_last else [3] + img_s
        if camera.rgb:
            elements.append(
                ObservationElement('%s_rgb' % prefix, shape, np.uint8))
        if camera.point_cloud:
            elements.append(
                ObservationElement('%s_point_cloud' % prefix, shape, np.float32))
            # elements.append(
            #     ObservationElement('%s_pixel_coord' % prefix, (2,), np.int32))
            elements.append(
                ObservationElement('%s_camera_extrinsics' % prefix, (4, 4),
                                   np.float32))
            elements.append(
                ObservationElement('%s_camera_intrinsics' % prefix, (3, 3),
                                   np.float32))
        if camera.mask:
            raise NotImplementedError()
        return elements

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

    def reset(self): # -> dict:
        self._record_current_episode = (
                self.eval and self._episode_index % RECORD_EVERY == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        self._i = 0
        self._prev_action = None
        descriptions, obs = self._task.reset()
        self._previous_obs = obs
        self._previous_obs_dict = self.extract_obs(obs)
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

    def step(self, act_result: ActResult): #-> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.
        try:
            obs, reward, terminal = self._task.step(action)
            if self._previous_obs is None:
                self._previous_obs = obs
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
        self._prev_action = action
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

# class MultiTaskRLBenchEnv:
#     """Stores a list of envs with different tasks"""
#     def __init__(self, train_tasks, eval_tasks, obs_config, action_mode, single_env_cfg):
#         self.env_list = OrderedDict()
#         self.task_names = {'train':[], 'eval': []}
#         for task in train_tasks:
#             env_cfg = deepcopy(single_env_cfg)
#             self.env_list[task] = CustomRLBenchEnv(
#                 observation_config=obs_config,
#                 action_mode=action_mode,
#                 task_class=task_file_to_task_class(task),
#                 **env_cfg)
#             print(f"Created Env for task {task}")
#             self.task_names['train'].append(task)
#         for task in eval_tasks:
#             self.task_names['eval'].append(task)
#             if task not in self.env_list.keys():
#                 print(f'Creating eval-only task: {task}')
#                 env_cfg = deepcopy(single_env_cfg)
#                 self.env_list[task] = CustomRLBenchEnv(
#                     observation_config=obs_config,
#                     action_mode=action_mode,
#                     task_class=task_file_to_task_class(task),
#                     **env_cfg)
             
#         self.n_train_tasks = len(self.task_names['train'])
#         self.n_eval_tasks = len(self.task_names['eval'])
#         self.n_unique_tasks = len( set(self.task_names['train'] + self.task_names['eval']) )
        
#         state_len = self.env_list[tasks[0]].low_dim_state_len
#         for env in self.env_list.values():
#             assert env.low_dim_state_len == state_len, 'Better if every env has the same state length '
#         self.low_dim_state_len = state_len 


#     def get_task_env(self, task):
#         assert task in self.env_list.keys(), f'Task {task} does not exist!'
#         return self.env_list[task]
    
#     def get_unique_tasks(self):
#         """Omits the overlap tasks between train and eval"""
#         return sorted( set(self.task_names['train'] + self.task_names['eval']) )