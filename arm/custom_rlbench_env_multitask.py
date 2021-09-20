from typing import Type, List

import numpy as np
from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy
from rlbench import ObservationConfig, CameraConfig
from rlbench.action_modes import ActionMode
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.task_environment import InvalidActionError

from yarr.agents.agent import ActResult, VideoSummary
from yarr.envs.rlbench_env import MultiTaskRLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

from collections import defaultdict

RECORD_EVERY = 20
from natsort import natsorted
from rlbench.backend.const import * # import those variables for get_demos()
from rlbench.utils import _resize_if_needed
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig
from PIL import Image 

import os  
from os import listdir
from os.path import join, exists
from typing import List
import pickle
from pyrep.objects import VisionSensor

class CustomMultiTaskRLBenchEnv(MultiTaskRLBenchEnv):
    """ add more stringent VideoSummary logging to save storage """
    def __init__(self,
                 task_classes: List[Type[Task]],
                 task_names: List[str],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 swap_task_every: int = 1,
                 reward_scale: int = 100,
                 num_video_limit: int = 3, # don't log too many videos of the same reward 
                 ):
        super(CustomMultiTaskRLBenchEnv, self).__init__(
            task_classes, task_names, 
            observation_config, action_mode,
            dataset_root, channels_last, headless, swap_task_every)
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False  
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._i = 0
        # self._num_video_limit = num_video_limit
        # self._logged_videos = defaultdict(int)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomMultiTaskRLBenchEnv, self).observation_elements
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

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)
        obs.gripper_matrix = grip_mat
        obs.gripper_pose = grip_pose
        return obs_dict

    def launch(self):
        super(CustomMultiTaskRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            self._record_current_episode = True 
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._previous_obs_dict = super(CustomMultiTaskRLBenchEnv, self).reset()
        self._record_current_episode = self.eval
        # self._record_current_episode = (
        #         self.eval and self._episode_index % RECORD_EVERY == 0)
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
        if ((terminal or self._i == self._episode_length) and self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            vid_name = f'task{self._active_task_id}_rollout/var{self._active_variation_id}_rew{int(reward)}'
            summaries.append( VideoSummary(vid_name, vid, fps=30) ) 
        return Transition(obs, reward, terminal, summaries=summaries) 

    def reset_to_demo(self, i):
        d, = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)
        self._task.reset_to_demo(d)

    def get_task(self, task_class):
        return self._rlbench_env.get_task(task_class)
        
    def custom_get_demos(self, 
                        task_name: str, 
                        amount: int,
                        variation_number=0,
                        image_paths=False,
                        random_selection: bool = True,
                        from_episode_number: int = 0,
                        load_cameras: List[str] = ['front_rgb'],
                        ) -> List[Demo]:
        """ only look for and load a subset of cameras , specified by load_cameras list
            modified from:
            - RLBench.rlbench.environment.get_demos and 
            - RLBench.rlbench.utils.get_stored_demos"""
        if self.env._dataset_root is None or len(self.env._dataset_root) == 0:
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")
        dataset_root = self.env._dataset_root
        obs_config = self.env._obs_config
        # basically changing the rlbench.utils.get_stored_demos method
        task_root = join(dataset_root, task_name)
        if not exists(task_root):
            raise RuntimeError("Can't find the demos for %s at: %s" % (
                task_name, task_root))
        # Sample an amount of examples for the variation of this task
        examples_path = join(
            task_root, VARIATIONS_FOLDER % variation_number,
            EPISODES_FOLDER)
        examples = listdir(examples_path)
        if amount == -1:
            amount = len(examples)
        if amount > len(examples):
            raise RuntimeError(
                'You asked for %d examples, but only %d were available.' % (
                    amount, len(examples)))
        if random_selection:
            selected_examples = np.random.choice(examples, amount, replace=False)
        else:
            selected_examples = natsorted(
                examples)   [from_episode_number:from_episode_number+amount]
            
        demos = []
        assert len(load_cameras) > 1, 'At least give one camera to look for'
        for example in selected_examples:
            example_path = join(examples_path, example)
            with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
                obs = pickle.load(f)

            l_rgb_f = join(example_path, LEFT_SHOULDER_RGB_FOLDER)
            l_depth_f = join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
            l_mask_f = join(example_path, LEFT_SHOULDER_MASK_FOLDER)
            r_rgb_f = join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
            r_depth_f = join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
            r_mask_f = join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
            oh_rgb_f = join(example_path, OVERHEAD_RGB_FOLDER)
            oh_depth_f = join(example_path, OVERHEAD_DEPTH_FOLDER)
            oh_mask_f = join(example_path, OVERHEAD_MASK_FOLDER)
            wrist_rgb_f = join(example_path, WRIST_RGB_FOLDER)
            wrist_depth_f = join(example_path, WRIST_DEPTH_FOLDER)
            wrist_mask_f = join(example_path, WRIST_MASK_FOLDER)
            front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
            front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)
            front_mask_f = join(example_path, FRONT_MASK_FOLDER)

            num_steps = len(obs)

            check_paths = [] # Okay if some camera folders don't exist 
            for cam in load_cameras:
                assert len(cam.split('_')) == 2, 'must specify camera names as xx_xx, e.g. front_rgb '
                pos, inp = cam.split('_')
                assert pos in ['l', 'r', 'oh', 'wrist', 'front'] and inp in ['rgb', 'depth', 'mask'], f"Got unsupported camera name {cam}"
                check_paths.append( locals()[f'{pos}_{inp}_f'] )
            
            for paths in check_paths:
                if num_steps != len( listdir(paths) ):
                    raise RuntimeError(f'Broken dataset assumption at dir {paths}')

            for i in range(num_steps):
                si = IMAGE_FORMAT % i
                if obs_config.left_shoulder_camera.rgb:
                    obs[i].left_shoulder_rgb = join(l_rgb_f, si)
                if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
                    obs[i].left_shoulder_depth = join(l_depth_f, si)
                if obs_config.left_shoulder_camera.mask:
                    obs[i].left_shoulder_mask = join(l_mask_f, si)
                if obs_config.right_shoulder_camera.rgb:
                    obs[i].right_shoulder_rgb = join(r_rgb_f, si)
                if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
                    obs[i].right_shoulder_depth = join(r_depth_f, si)
                if obs_config.right_shoulder_camera.mask:
                    obs[i].right_shoulder_mask = join(r_mask_f, si)
                if obs_config.overhead_camera.rgb:
                    obs[i].overhead_rgb = join(oh_rgb_f, si)
                if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
                    obs[i].overhead_depth = join(oh_depth_f, si)
                if obs_config.overhead_camera.mask:
                    obs[i].overhead_mask = join(oh_mask_f, si)
                if obs_config.wrist_camera.rgb:
                    obs[i].wrist_rgb = join(wrist_rgb_f, si)
                if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                    obs[i].wrist_depth = join(wrist_depth_f, si)
                if obs_config.wrist_camera.mask:
                    obs[i].wrist_mask = join(wrist_mask_f, si)
                if obs_config.front_camera.rgb:
                    obs[i].front_rgb = join(front_rgb_f, si)
                if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                    obs[i].front_depth = join(front_depth_f, si)
                if obs_config.front_camera.mask:
                    obs[i].front_mask = join(front_mask_f, si)

                # Remove low dim info if necessary
                if not obs_config.joint_velocities:
                    obs[i].joint_velocities = None
                if not obs_config.joint_positions:
                    obs[i].joint_positions = None
                if not obs_config.joint_forces:
                    obs[i].joint_forces = None
                if not obs_config.gripper_open:
                    obs[i].gripper_open = None
                if not obs_config.gripper_pose:
                    obs[i].gripper_pose = None
                if not obs_config.gripper_joint_positions:
                    obs[i].gripper_joint_positions = None
                if not obs_config.gripper_touch_forces:
                    obs[i].gripper_touch_forces = None
                if not obs_config.task_low_dim_state:
                    obs[i].task_low_dim_state = None

            if not image_paths:
                for i in range(num_steps):
                    if obs_config.left_shoulder_camera.rgb:
                        obs[i].left_shoulder_rgb = np.array(
                            _resize_if_needed(
                                Image.open(obs[i].left_shoulder_rgb),
                                obs_config.left_shoulder_camera.image_size))
                    if obs_config.right_shoulder_camera.rgb:
                        obs[i].right_shoulder_rgb = np.array(
                            _resize_if_needed(Image.open(
                            obs[i].right_shoulder_rgb),
                                obs_config.right_shoulder_camera.image_size))
                    if obs_config.overhead_camera.rgb:
                        obs[i].overhead_rgb = np.array(
                            _resize_if_needed(Image.open(
                            obs[i].overhead_rgb),
                                obs_config.overhead_camera.image_size))
                    if obs_config.wrist_camera.rgb:
                        obs[i].wrist_rgb = np.array(
                            _resize_if_needed(
                                Image.open(obs[i].wrist_rgb),
                                obs_config.wrist_camera.image_size))
                    if obs_config.front_camera.rgb:
                        obs[i].front_rgb = np.array(
                            _resize_if_needed(
                                Image.open(obs[i].front_rgb),
                                obs_config.front_camera.image_size))

                    if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
                        l_sh_depth = image_to_float_array(
                            _resize_if_needed(
                                Image.open(obs[i].left_shoulder_depth),
                                obs_config.left_shoulder_camera.image_size),
                            DEPTH_SCALE)
                        near = obs[i].misc['left_shoulder_camera_near']
                        far = obs[i].misc['left_shoulder_camera_far']
                        l_sh_depth_m = near + l_sh_depth * (far - near)
                        if obs_config.left_shoulder_camera.depth:
                            d = l_sh_depth_m if obs_config.left_shoulder_camera.depth_in_meters else l_sh_depth
                            obs[i].left_shoulder_depth = obs_config.left_shoulder_camera.depth_noise.apply(d)
                        else:
                            obs[i].left_shoulder_depth = None

                    if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
                        r_sh_depth = image_to_float_array(
                            _resize_if_needed(
                                Image.open(obs[i].right_shoulder_depth),
                                obs_config.right_shoulder_camera.image_size),
                            DEPTH_SCALE)
                        near = obs[i].misc['right_shoulder_camera_near']
                        far = obs[i].misc['right_shoulder_camera_far']
                        r_sh_depth_m = near + r_sh_depth * (far - near)
                        if obs_config.right_shoulder_camera.depth:
                            d = r_sh_depth_m if obs_config.right_shoulder_camera.depth_in_meters else r_sh_depth
                            obs[i].right_shoulder_depth = obs_config.right_shoulder_camera.depth_noise.apply(d)
                        else:
                            obs[i].right_shoulder_depth = None

                    if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
                        oh_depth = image_to_float_array(
                            _resize_if_needed(
                                Image.open(obs[i].overhead_depth),
                                obs_config.overhead_camera.image_size),
                            DEPTH_SCALE)
                        near = obs[i].misc['overhead_camera_near']
                        far = obs[i].misc['overhead_camera_far']
                        oh_depth_m = near + oh_depth * (far - near)
                        if obs_config.overhead_camera.depth:
                            d = oh_depth_m if obs_config.overhead_camera.depth_in_meters else oh_depth
                            obs[i].overhead_depth = obs_config.overhead_camera.depth_noise.apply(d)
                        else:
                            obs[i].overhead_depth = None

                    if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                        wrist_depth = image_to_float_array(
                            _resize_if_needed(
                                Image.open(obs[i].wrist_depth),
                                obs_config.wrist_camera.image_size),
                            DEPTH_SCALE)
                        near = obs[i].misc['wrist_camera_near']
                        far = obs[i].misc['wrist_camera_far']
                        wrist_depth_m = near + wrist_depth * (far - near)
                        if obs_config.wrist_camera.depth:
                            d = wrist_depth_m if obs_config.wrist_camera.depth_in_meters else wrist_depth
                            obs[i].wrist_depth = obs_config.wrist_camera.depth_noise.apply(d)
                        else:
                            obs[i].wrist_depth = None

                    if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                        front_depth = image_to_float_array(
                            _resize_if_needed(
                                Image.open(obs[i].front_depth),
                                obs_config.front_camera.image_size),
                            DEPTH_SCALE)
                        near = obs[i].misc['front_camera_near']
                        far = obs[i].misc['front_camera_far']
                        front_depth_m = near + front_depth * (far - near)
                        if obs_config.front_camera.depth:
                            d = front_depth_m if obs_config.front_camera.depth_in_meters else front_depth
                            obs[i].front_depth = obs_config.front_camera.depth_noise.apply(d)
                        else:
                            obs[i].front_depth = None

                    if obs_config.left_shoulder_camera.point_cloud:
                        obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                            l_sh_depth_m,
                            obs[i].misc['left_shoulder_camera_extrinsics'],
                            obs[i].misc['left_shoulder_camera_intrinsics'])
                    if obs_config.right_shoulder_camera.point_cloud:
                        obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                            r_sh_depth_m,
                            obs[i].misc['right_shoulder_camera_extrinsics'],
                            obs[i].misc['right_shoulder_camera_intrinsics'])
                    if obs_config.overhead_camera.point_cloud:
                        obs[i].overhead_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                            oh_depth_m,
                            obs[i].misc['overhead_camera_extrinsics'],
                            obs[i].misc['overhead_camera_intrinsics'])
                    if obs_config.wrist_camera.point_cloud:
                        obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                            wrist_depth_m,
                            obs[i].misc['wrist_camera_extrinsics'],
                            obs[i].misc['wrist_camera_intrinsics'])
                    if obs_config.front_camera.point_cloud:
                        obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                            front_depth_m,
                            obs[i].misc['front_camera_extrinsics'],
                            obs[i].misc['front_camera_intrinsics'])

                    # Masks are stored as coded RGB images.
                    # Here we transform them into 1 channel handles.
                    if obs_config.left_shoulder_camera.mask:
                        obs[i].left_shoulder_mask = rgb_handles_to_mask(
                            np.array(_resize_if_needed(Image.open(
                                obs[i].left_shoulder_mask),
                                obs_config.left_shoulder_camera.image_size)))
                    if obs_config.right_shoulder_camera.mask:
                        obs[i].right_shoulder_mask = rgb_handles_to_mask(
                            np.array(_resize_if_needed(Image.open(
                                obs[i].right_shoulder_mask),
                                obs_config.right_shoulder_camera.image_size)))
                    if obs_config.overhead_camera.mask:
                        obs[i].overhead_mask = rgb_handles_to_mask(
                            np.array(_resize_if_needed(Image.open(
                                obs[i].overhead_mask),
                                obs_config.overhead_camera.image_size)))
                    if obs_config.wrist_camera.mask:
                        obs[i].wrist_mask = rgb_handles_to_mask(np.array(
                            _resize_if_needed(Image.open(
                                obs[i].wrist_mask),
                                obs_config.wrist_camera.image_size)))
                    if obs_config.front_camera.mask:
                        obs[i].front_mask = rgb_handles_to_mask(np.array(
                            _resize_if_needed(Image.open(
                                obs[i].front_mask),
                                obs_config.front_camera.image_size)))

            demos.append(obs)
        return demos