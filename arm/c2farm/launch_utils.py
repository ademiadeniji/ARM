import logging
from typing import List

import numpy as np
from omegaconf import DictConfig
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from yarr.envs.env import Env
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer, ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from arm import demo_loading_utils, utils
from arm.custom_rlbench_env import CustomRLBenchEnv
from arm.preprocess_agent import PreprocessAgent
from arm.c2farm.networks import Qattention3DNet, Qattention3DNetWithContext, Qattention3DNetWithFiLM
from arm.c2farm.qattention_agent import QAttentionAgent
from arm.c2farm.qattention_agent_with_context import QAttentionContextAgent
from arm.c2farm.qattention_stack_agent import QAttentionStackAgent, QAttentionStackContextAgent
from arm.c2farm.context_agent import ContextAgent
from arm.c2farm.discrete_context_agent import DiscreteContextAgent
from arm.network_utils import SiameseCNNWithFCModel

from arm.models.slowfast  import TempResNet
from arm.demo_dataset import MultiTaskDemoSampler, RLBenchDemoDataset, collate_by_id
from arm.models.utils import make_optimizer 
from functools import partial
from torch.utils.data import DataLoader

REWARD_SCALE = 100.0
TASK_ID='task_id'
VAR_ID='variation_id'
CONTEXT_SIZE=64


def create_replay(batch_size: int, timesteps: int, prioritisation: bool,
                  save_dir: str, cameras: list, env: Env,
                  voxel_sizes, replay_size=1e5):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)

    observation_elements = env.observation_elements
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_pixel_coord' % cname, (2,), np.int32))
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.int32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.int32)
    ])

    for depth in range(len(voxel_sizes)):
        observation_elements.append(
            ReplayElement('attention_coordinate_layer_%d' % depth, (3,), np.float32)
        )

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
        ReplayElement('task_id', (), np.uint8),
        #ReplayElement('task_name', (), str),
        ReplayElement('variation_id', (), np.uint8),
    ]

    replay_class = UniformReplayBuffer
    if prioritisation:
        replay_class = PrioritizedReplayBuffer
    replay_buffer = replay_class(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer

def _get_action(
        obs_tp1: Observation,
        rlbench_scene_bounds: List[float],   # AKA: DEPTH0_BOUNDS
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    assert len(bounds_offset) == len(voxel_sizes) -1
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    for depth, vox_size in enumerate(voxel_sizes):
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                     attention_coordinate + bounds_offset[depth - 1]])
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates

def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        env: CustomRLBenchEnv,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],   # AKA: DEPTH0_BOUNDS
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        task: str,
        variation: int, 
        task_id: int, 
        ):
    prev_action = None
    obs = inital_obs
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        trans_indicies, rot_grip_indicies, action, attention_coordinates = _get_action(
            obs_tp1, rlbench_scene_bounds, voxel_sizes, bounds_offset,
            rotation_resolution, crop_augmentation)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = env.extract_obs(obs, t=k, prev_action=prev_action)
        prev_action = np.copy(action)

        others = {'demo': True, TASK_ID: task_id , VAR_ID: variation}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
        }

        for depth in range(len(voxel_sizes)):
            final_obs['attention_coordinate_layer_%d' % depth] = \
                attention_coordinates[depth]
        for name in cameras:
            px, py = utils.point_to_pixel_index(
                obs_tp1.gripper_pose[:3],
                obs_tp1.misc['%s_camera_extrinsics' % name],
                obs_tp1.misc['%s_camera_intrinsics' % name])
            final_obs['%s_pixel_coord' % name] = [py, px]
        others.update(final_obs)
        others.update(obs_dict)
         
        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1  # Set the next obs

    # Final step
    obs_dict_tp1 = env.extract_obs(
        obs_tp1, t=k + 1, prev_action=prev_action)
    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)

def fill_replay(replay: ReplayBuffer,
                task: str,
                env: CustomRLBenchEnv,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                variation: int = 0,
                task_id: int = 0,
                ):

    logging.info(f'Task {task} variation#{variation} Filling replay with {num_demos} demos...')
    for d_idx in range(num_demos):
        # NOTE(Mandi): changed here to only load fron cameras, to save local dataset size
        # demo = env.env.get_demos(
        #     task, 1, variation_number=variation, random_selection=False,
        #     from_episode_number=d_idx)[0]
        lookup_cams = []
        for cam in cameras: # must be in [l, r, oh, wrist, front,]
            lookup_cams.extend([cam + '_rgb', cam + '_depth'])
        demo = env.custom_get_demos(
            task, 1, variation_number=variation, random_selection=False,
            from_episode_number=d_idx, load_cameras=lookup_cams)[0]
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo)
        # print(f'Loading demo idx {d_idx}: found keypoints', episode_keypoints)

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue
            initial_obs = demo[i]
            # If our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            # print(f'Loading demo idx {d_idx}: adding tstep {i}, keypoints', episode_keypoints)
            _add_keypoints_to_replay(
                replay, initial_obs, demo, env, episode_keypoints, cameras,
                rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, task, variation, task_id)
    #logging.info('Replay filled with demos.')

def create_agent(cfg: DictConfig, env, depth_0bounds=None, cam_resolution=None):
    VOXEL_FEATS = 3
    LATENT_SIZE = 64
    depth_0bounds = depth_0bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    cam_resolution = cam_resolution or [128, 128]

    include_prev_layer = False

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        if depth == 0:
            unet3d = Qattention3DNet(
                in_channels=VOXEL_FEATS + 3 + 1 + 3,
                out_channels=1,
                voxel_size=vox_size,
                out_dense=0,
                kernels=LATENT_SIZE,
                norm=None if 'None' in cfg.method.norm else cfg.method.norm,
                dense_feats=128,
                activation=cfg.method.activation,
                low_dim_size=env.low_dim_state_len)
        else:
            last = depth == len(cfg.method.voxel_sizes) - 1
            unet3d = Qattention3DNet(
                in_channels=VOXEL_FEATS + 3 + 1 + 3,
                out_channels=2,
                voxel_size=vox_size,
                out_dense=(num_rotation_classes * 3) if last else 0,
                kernels=LATENT_SIZE,
                dense_feats=128,
                norm=None if 'None' in cfg.method.norm else cfg.method.norm,
                activation=cfg.method.activation,
                low_dim_size=env.low_dim_state_len,
                include_prev_layer=include_prev_layer)

        qattention_agent = QAttentionAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            unet3d=unet3d,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            tau=cfg.method.tau,
            lr=cfg.method.lr,
            lambda_trans_qreg=cfg.method.lambda_trans_qreg,
            lambda_rot_qreg=cfg.method.lambda_rot_qreg,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size if cfg.replay.share_across_tasks else cfg.replay.total_batch_size,
            voxel_feature_size=3,
            exploration_strategy=cfg.method.exploration_strategy,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            grad_clip=0.01,
            gamma=0.99,
            q_thres=cfg.dev.q_thres if cfg.get('dev', None) else 0.75,
        )
        qattention_agents.append(qattention_agent)

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )
    preprocess_agent = PreprocessAgent(pose_agent=rotation_agent)
    return preprocess_agent

def create_agent_with_context(cfg: DictConfig, env, 
        depth_0bounds=None, cam_resolution=None):
    """let's keep the hinge/representation loss completely separate from 
        replay_sampled used for Q-attention updates: have an context_agent to calculate 
        embeddings and embedding losses, but a separate update() function that only takes in
        context inputs (i.e. no actions/terminal/other info); context_agent also has an act()
        method, only this one takes in replay_samples and produce embeddings used for q-attention
        agent, and the optimizers in qattention agents optionally also updates context_agent's embedder
        """
    VOXEL_FEATS = 3
    LATENT_SIZE = 64
    depth_0bounds = depth_0bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    cam_resolution = cam_resolution or [128, 128]
    include_prev_layer = False
    
    # New(Mandi): create context agent here 
    # embedding_net = SiameseCNNWithFCModel(
    #     input_shapes=[
    #         [(3 * embedding_timesteps) + 
    #         (train_env.action_shape[0] * embedding_timesteps if cfg.contexts.with_action else 0),
    #         cam_resolution[0], cam_resolution[1]]
    #             ] * 2,
    #     filters=[64, 64, 64],
    #     kernel_sizes=[3, 3, 3],
    #     strides=[2, 2, 2],
    #     pre_filters=[32],
    #     pre_kernel_sizes=[3],
    #     pre_strides=[1],
    #     norm=None if 'None' in cfg.method.norm else cfg.method.norm,
    #     activation=cfg.method.activation, # same as c2farm
    #     fc_layers=[64, 64, embedding_size]
    #     )  
     
    if cfg.dev.discrete:
        logging.info('Using discrete embedding context!')   
        context_agent = DiscreteContextAgent(  
            one_hot=(cfg.dev.one_hot or cfg.dev.noisy_one_hot),
            replay_update_freq=cfg.dev.replay_update_freq,
            **cfg.contexts.discrete_agent
            )             
    else:
        embedding_net = TempResNet(cfg.encoder) 
        context_agent = ContextAgent(
            embedding_net=embedding_net, 
            camera_names=cfg.rlbench.cameras,
            one_hot=(cfg.dev.one_hot or cfg.dev.noisy_one_hot),
            replay_update_freq=cfg.dev.replay_update_freq,
            **cfg.contexts.agent
            )

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    if cfg.dev.one_hot:
        ctxt_size = sum([len(variations) for variations in cfg.rlbench.all_variations]) 
    elif cfg.dev.noisy_one_hot:
        ctxt_size = sum([len(variations) for variations in cfg.rlbench.use_variations]) 
    elif cfg.dev.noisy_dim_20:
        ctxt_size = 20
    else: 
        ctxt_size = cfg.contexts.agent.embedding_size * 4 
        if cfg.dev.discrete:
            ctxt_size = 256
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        if depth == 0:
            if cfg.dev.use_film:
                unet3d = Qattention3DNetWithFiLM(
                    in_channels=VOXEL_FEATS + 3 + 1 + 3,
                    out_channels=1,
                    voxel_size=vox_size,
                    out_dense=0,
                    kernels=LATENT_SIZE,
                    norm=None if 'None' in cfg.method.norm else cfg.method.norm,
                    dense_feats=128,
                    activation=cfg.method.activation,
                    low_dim_size=env.low_dim_state_len,
                    inp_context_size=ctxt_size, 
                    use_context=True, 
                    dev_cfgs=dict(cfg.dev),
                )
            else:
                unet3d = Qattention3DNetWithContext(
                in_channels=VOXEL_FEATS + 3 + 1 + 3,
                out_channels=1,
                voxel_size=vox_size,
                out_dense=0,
                kernels=LATENT_SIZE,
                norm=None if 'None' in cfg.method.norm else cfg.method.norm,
                dense_feats=128,
                activation=cfg.method.activation,
                low_dim_size=env.low_dim_state_len,
                inp_context_size=ctxt_size,
                use_context=True, 
                encode_context=cfg.dev.encode_context,
                encode_context_size=cfg.dev.qnet_context_latent_size,
                dev_cfgs=dict(cfg.dev),
                )
        else:
            last = depth == len(cfg.method.voxel_sizes) - 1 
            if cfg.dev.use_film:
                unet3d = Qattention3DNetWithFiLM(
                    in_channels=VOXEL_FEATS + 3 + 1 + 3,
                    out_channels=2,
                    voxel_size=vox_size,
                    out_dense=(num_rotation_classes * 3) if last else 0,
                    kernels=LATENT_SIZE,
                    dense_feats=128,
                    norm=None if 'None' in cfg.method.norm else cfg.method.norm,
                    activation=cfg.method.activation,
                    low_dim_size=env.low_dim_state_len,
                    include_prev_layer=include_prev_layer,
                    use_context=(not cfg.dev.single_layer_context), 
                    inp_context_size=cfg.dev.qnet_context_latent_size if (cfg.contexts.pass_down_context and cfg.dev.encode_context) else ctxt_size, #cfg.contexts.agent.embedding_size,
                    dev_cfgs=dict(cfg.dev),
                    )
            else:
                unet3d = Qattention3DNetWithContext(
                    in_channels=VOXEL_FEATS + 3 + 1 + 3,
                    out_channels=2,
                    voxel_size=vox_size,
                    out_dense=(num_rotation_classes * 3) if last else 0,
                    kernels=LATENT_SIZE,
                    dense_feats=128,
                    norm=None if 'None' in cfg.method.norm else cfg.method.norm,
                    activation=cfg.method.activation,
                    low_dim_size=env.low_dim_state_len,
                    include_prev_layer=include_prev_layer,
                    use_context=(not cfg.dev.single_layer_context),
                    encode_context=cfg.dev.encode_context,
                    inp_context_size=cfg.dev.qnet_context_latent_size if (cfg.contexts.pass_down_context and cfg.dev.encode_context) else ctxt_size, #cfg.contexts.agent.embedding_size,
                    encode_context_size=cfg.dev.qnet_context_latent_size,
                    dev_cfgs=dict(cfg.dev),
                    )

        qattention_agent = QAttentionContextAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            unet3d=unet3d,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            tau=cfg.method.tau,
            lr=cfg.method.lr,
            emb_lr=cfg.method.emb_lr,
            lambda_trans_qreg=cfg.method.lambda_trans_qreg,
            lambda_rot_qreg=cfg.method.lambda_rot_qreg,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size if cfg.replay.share_across_tasks else cfg.replay.total_batch_size,
            voxel_feature_size=3,
            exploration_strategy=cfg.method.exploration_strategy,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            grad_clip=0.01,
            gamma=0.99,
            context_agent=context_agent, 
            update_context_agent=(cfg.dev.qagent_update_context and depth == 0),
            pass_down_context=cfg.contexts.pass_down_context,
            use_emb_loss=cfg.dev.qagent_use_emb_loss, 
            emb_weight=cfg.contexts.emb_weight,
            one_hot=(cfg.dev.one_hot or cfg.dev.noisy_one_hot),
        )
        qattention_agents.append(qattention_agent)

    rotation_agent = QAttentionStackContextAgent(
        context_agent=context_agent,
        pass_down_context=cfg.contexts.pass_down_context,
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
        rotation_prediction_depth=0,
    )
    preprocess_agent = PreprocessAgent(
        pose_agent=rotation_agent,
        context_agent=context_agent
        )
    return preprocess_agent
