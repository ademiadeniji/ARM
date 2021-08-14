"""New(0805)"""
import logging

import numpy as np
import pyrender
import torch
from absl import app
from hydra.experimental import initialize, compose
from moviepy.editor import *
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from omegaconf import OmegaConf, ListConfig, DictConfig
from rlbench.action_modes import ActionMode, ArmActionMode, GripperActionMode
from rlbench.backend.utils import task_file_to_task_class

from arm import c2farm
from arm.custom_rlbench_env import CustomRLBenchEnv
from arm.utils import visualise_voxel
from launch_multitask import _create_obs_config
# from tools.utils import RLBenchCinematic
from os.path import join 
import hydra 
from glob import glob 
FREEZE_DURATION = 2
FPS = 20


def _create_voxel_animation(voxel_grid, q, voxel_idx, steps=100):
    alpha = 1.0
    rot = 0.0
    alpha_delta = -0.1
    rot_delta = -0.02
    frames = []
    r = pyrender.OffscreenRenderer(
        viewport_width=640, viewport_height=480, point_size=1.0)
    for s in range(steps):
        alpha += alpha_delta
        rot += rot_delta
        alpha = float(np.clip(alpha, 0, 1))
        if alpha <= 0 or alpha >= 1:
            alpha_delta = -alpha_delta  # flip
        frames.append(visualise_voxel(
            voxel_grid, q, voxel_idx, highlight_alpha=alpha,
            rotation_amount=rot % (2*np.pi),
            offscreen_renderer=r))
    return frames


def _clip_with_text(clip, text: str, duration: int = None):
    # If this fails, see: https://github.com/Zulko/moviepy/issues/693
    txt_clip = TextClip(text, fontsize=30, color='black')
    txt_clip = txt_clip.set_pos('bottom').set_duration(duration or clip.duration)
    return CompositeVideoClip([clip, txt_clip])


def _mix_clips(scene_frames: list, voxel_frames: list, large_voxel_windows: bool = False, white=False, step=0, reward=0, episode=0):

    scene_clip = ImageSequenceClip(scene_frames, fps=FPS)
    scene_freeze = scene_clip.to_ImageClip(0).set_duration(FREEZE_DURATION)
    scene_clip = concatenate_videoclips([scene_freeze, scene_clip])
    scene_clip = _clip_with_text(scene_clip, 'Scene', duration=scene_clip.duration)
    clips = [scene_clip]
    for i, f in enumerate(voxel_frames):
        if white:
            f = [np.ones_like(ff) + 255 for ff in f]
        c = ImageSequenceClip(f, fps=FPS)
        if not white:
            c = _clip_with_text(c, 'Depth %d' % i + ' Eps %d'%episode + ' Step %d' % step + ' Reward %d' % reward )
        img = c.to_ImageClip(c.duration-0.1).set_duration(5)
        c = concatenate_videoclips([c, img]).subclip(0, scene_clip.duration)
        clips.append(c)

    if not large_voxel_windows:
        ca = clips_array(
            [[c.resize(1.0 / float(len(voxel_frames)))] for c in clips[1:]])
        clips = [clips[0], ca]

    joined_clip = clips_array([clips])
    # joined_clip.preview()
    return joined_clip


def _save_clips(clips, name):
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile('%s.mp4' % name)


def visualise(cfg: DictConfig):
    
    resume_dir = join(cfg.resume_path, cfg.resume_run, 'weights', str(cfg.resume_step))
    print(cfg, resume_dir)
    if not os.path.exists(resume_dir):
        print('Cannot find the weights saved at path: '+resume_dir)
        
        resume_dir = sorted( glob( join(cfg.resume_path, cfg.resume_run, 'weights/*') ) ) [-1]
        print('Default to using last saved checkpoint: ', resume_dir)

    config_path = join(cfg.resume_path, cfg.resume_run, 'config.yaml')
    loaded_cfg = OmegaConf.load(config_path)
    # if not os.path.exists(config_path):
    #     raise ValueError('No cofig in: ' + config_path)
    # if not os.path.exists(weights_path):
    #     raise ValueError('No weights in: ' + weights_path)

    # with initialize(config_path=os.path.relpath(config_path)):
    #     cfg = compose(config_name="config")
    # print(OmegaConf.to_yaml(cfg))

    cfg.rlbench.cameras = ['front']

    obs_config = _create_obs_config(
        cfg.rlbench.cameras, loaded_cfg.rlbench.camera_resolution)
    
    task = cfg.tasks[0]
    task_class = task_file_to_task_class(cfg.tasks[0])
    action_mode = ActionMode(
        ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME_WITH_COLLISION_CHECK,
        GripperActionMode.OPEN_AMOUNT)
    env = CustomRLBenchEnv(
        task_class=task_class, observation_config=obs_config,
        action_mode=action_mode, dataset_root='',
        episode_length=loaded_cfg.rlbench.episode_length, headless=True)
    _ = env.observation_elements

    
    agent = c2farm.launch_utils.create_agent(loaded_cfg, env)
    agent.build(training=False, device=torch.device("cpu"))

 
    agent.load_weights(resume_dir)
    logging.info('Launching env with task(s): {}'.format(cfg.tasks))
    
    clips = []
    avg_rewards = []
    for ep in range(cfg.episodes):
        env.eval = True 
        env.launch()
        cinemtaic_cam = env._record_cam #RLBenchCinematic()  
        env.register_callback(env._my_callback)
        env._record_current_episode = True 
        obs = env.reset()
        # trajectory_frames = [ (cinemtaic_cam.capture_rgb() * 255).astype(np.uint8)] #cinemtaic_cam.frames
        # cinemtaic_cam.handle_explicitly()

        agent.reset()
        obs_history = {
            k: [np.array(v, dtype=_get_type(v))] * loaded_cfg.replay.timesteps for k, v in obs.items()}
        
        last = False
        num_past_frame = 0
        for step in range(loaded_cfg.rlbench.episode_length):
            
            prepped_data = {k: np.array([v]) for k, v in obs_history.items()}
            act_result = agent.act(step, prepped_data, deterministic=True)
            transition = env.step(act_result)

            #trajectory_frames = [ (cinemtaic_cam.capture_rgb() * 255).astype(np.uint8)] #cinemtaic_cam.frames
            trajectory_frames = env._recorded_images
            print('Env recorded number of frames:', len(trajectory_frames))
            ### 
            # if len(trajectory_frames) == 0:
            #     frames = [ (cinemtaic_cam.capture_rgb() * 255).astype(np.uint8)] 
            #     voxel_depths = []
            #     d = 0
            #     while True:
            #         if 'voxel_grid_depth%d' % d not in act_result.info:
            #             break
            #         logging.info('Episode step: %d. Creating voxel animation '
            #                      'for depth: %d' % (step, d))
            #         voxel_depths.append(_create_voxel_animation(
            #             act_result.info['voxel_grid_depth%d' % d].numpy()[0],
            #             act_result.info['q_depth%d' % d].numpy()[0],
            #             act_result.info['voxel_idx_depth%d' % d].numpy()[0],
            #         ))
            #         d += 1

            #     # if transition.terminal:
            #     #     last = True 
                    
            #     clips.append(_mix_clips(frames, 
            #         voxel_depths, white=last, step=step, reward=transition.reward, episode=ep))


            if len(trajectory_frames)  >  num_past_frame: # there's new frames being logged
                # cinemtaic_cam.empty()
                logging.info('Episode step: %d. Mixing clips ' % step) 
                voxel_depths = []
                d = 0
                while True:
                    if 'voxel_grid_depth%d' % d not in act_result.info:
                        break
                    # logging.info('Episode step: %d. Creating voxel animation '
                    #              'for depth: %d' % (step, d))
                    voxel_depths.append(_create_voxel_animation(
                        act_result.info['voxel_grid_depth%d' % d].numpy()[0],
                        act_result.info['q_depth%d' % d].numpy()[0],
                        act_result.info['voxel_idx_depth%d' % d].numpy()[0],
                    ))
                    d += 1

                # if transition.terminal:
                #     last = True 
                    
                clips.append(_mix_clips(trajectory_frames[num_past_frame:], 
                    voxel_depths, white=last, step=step, reward=transition.reward, episode=ep))

            #env._recorded_images.clear()
            num_past_frame = len(trajectory_frames)
            if last:
                break
            if transition.terminal:
                #cinemtaic_cam.handle_explicitly()
                last = True
                logging.info('Episode done, reward:{}'.format(transition.reward))
                avg_rewards.append(transition.reward)
            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)
    avg_r = np.mean(avg_rewards)
    _save_clips(clips, '/home/mandi/ARM/{}_rew{}_episode{}_c2f_qattention_{}'.format(cfg.vid_name, avg_r, len(avg_rewards), task))

    print('Shutting down env...')
    env.shutdown()


def _get_type(x):
    if x.dtype == np.float64:
        return np.float32
    return x.dtype


@hydra.main(config_name='video', config_path='conf')
def main(cfg: DictConfig) -> None:
    
    visualise(cfg)


if __name__ == '__main__':
    main()