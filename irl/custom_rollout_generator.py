from multiprocessing import Value
import os 
import numpy as np
from typing import Any, List, Union 
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.envs.env import MultiTaskEnv
from yarr.envs.rlbench_env import MultiTaskRLBenchEnv
from yarr.utils.transition import ReplayTransition
from models import RewardMLP
import torch 
import torch.nn.functional as F
import clip 
from PIL import Image

from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
try:
    from torchvision.transforms import InterpolationMode 
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from einops import rearrange
CONTEXT_KEY = 'demo_sample' 
DEMO_KEY='front_rgb'
TASK_ID='task_id'
VAR_ID='variation_id'
CHECKPT='agent_checkpoint'

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    

class CustomRolloutGenerator(object):
    """For each env step, also sample from the demo dataset to 
        generate context embeddings"""

    def __init__(
        self,  
        sample_key='front_rgb', 
        one_hot=False, 
        noisy_one_hot=False,
        num_task_vars=20,
        task_var_to_replay_idx=dict(),
        dev_cfg={}, 
        rew_cfg={},
        replay_buffers=None,
        ):
        self._sample_key = sample_key 
        self._one_hot = one_hot 
        self._noisy_one_hot = noisy_one_hot 
        self._num_task_vars = num_task_vars 
        self._task_var_to_replay_idx = task_var_to_replay_idx
        self._dev_cfg = dev_cfg
        self._rew_cfg = rew_cfg
 
        self._replays  = replay_buffers
        print('Loading CLIP checkpoint!')
        self._clip_model, clip_process = clip.load('ViT-L/14', device='cuda:1') 
        model_name = os.path.join(self._rew_cfg.model_path, self._rew_cfg.model, f'{rew_cfg.step}.pt')
        self._reward_model = RewardMLP(self._clip_model, self._rew_cfg.predict_logits).to('cuda:1')
        self._reward_model.load_state_dict(torch.load(model_name))

        self._text = clip.tokenize([str(self._rew_cfg.prompts)]).to('cuda:1')
        # self._text_features = self_clip_model.encode_text(text)
        if self._rew_cfg.use_aug:
            self._aug_process = Compose([
                Resize(240, interpolation=BICUBIC),
                RandomResizedCrop(224, scale=(0.7, 1.0)), 
                RandomHorizontalFlip(0.5), 
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self._aug_process = clip_process

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
 
    
    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool, 
                  swap_task: bool = True, ): 
        obs     = env.reset(swap_task=swap_task) 
        task_id = env._active_task_id
        variation_id = env._active_variation_id
        task_name = env._active_task_name
        buf_id = self._task_var_to_replay_idx[task_id][variation_id]
        #print('mt rollout gen:', task_id, variation_id, task_name)
        agent.reset()
        checkpoint = agent.get_checkpoint()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        demo_samples = None
        one_hot_vec = F.one_hot(  torch.tensor(int(buf_id)), num_classes=self._num_task_vars)
        one_hot_vec = one_hot_vec.clone().detach().to(torch.float32)
        episode_trans = [] 
         
        for step in range(episode_length): 
            prepped_data = {k: np.array([v]) for k, v in obs_history.items()}
            prepped_data.update({
                TASK_ID: task_id, 
                VAR_ID: variation_id})
             
            # print('rollout generator input:', prepped_data.keys())


            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            agent_extra_elems = {k: np.array(v) for k, v in
                                 act_result.replay_elements.items()}

            transition = env.step(act_result)
            assert env._active_task_name == task_name and env._active_task_id == task_id and env._active_variation_id == variation_id, \
                 'Something is wrong with RLBench Env, task {task_name} is replaced by {env.active_task_name} in middle of an episode'
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs.update(agent_obs_elems)
            # obs.update(agent_extra_elems)
            obs_tp1 = dict(transition.observation)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info.update({
                TASK_ID: task_id, 
                VAR_ID: variation_id, #'task_name': env._active_task_name
                'demo': False,
                CHECKPT: checkpoint
                })

            replay_transition = ReplayTransition(
                observation=obs, action=act_result.action, reward=transition.reward,
                terminal=transition.terminal, timeout=timeout,  
                info=transition.info,
                summaries=transition.summaries,)

            # if transition.terminal:
            #     print('rollout gen got transition:', transition.summaries)
                  
            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: np.array([v]) for k, v in
                                    obs_history.items()} 

                    prepped_data.update({
                        TASK_ID: task_id, 
                        VAR_ID: variation_id})
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            obs = dict(transition.observation)
            episode_trans.append(replay_transition)
            # yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                # return
                break 

        episode_success = episode_trans[-1].reward > 0 
        episode_trans[-1].info['task_success'] = episode_success
        if not eval:
            use_scale = self._rew_cfg.scale_logits
            obs = [ep.observation['front_rgb'] for ep in episode_trans]
            if self._rew_cfg.shift_t:
                # label with next obs's rew!
                obs = obs[1:] + [episode_trans[-1].final_observation['front_rgb']]
            obs = [Image.fromarray(np.uint8(ob.transpose((1,2,0)))).convert('RGB') for ob in obs]
            if self._rew_cfg.use_aug and self._rew_cfg.aug_avg > 1:
                itrs = int(self._rew_cfg.aug_avg)
                logits = 0
                for _ in range(itrs):
                    imgs = torch.stack([self._aug_process(ob) for ob in obs]).to('cuda:1')
                    logits += self._reward_model(imgs, self._text, scale_logits=use_scale).detach().cpu().numpy()
                logits /= itrs
            else:
                imgs = torch.stack([self._aug_process(ob) for ob in obs]).to('cuda:1') 
                logits = self._reward_model(imgs, self._text, scale_logits=use_scale).detach().cpu().numpy()

            # if self._reward_model.predict_logits:
            #     logits *= 50 # reward is ~[-1.3*50, 1.8*50]
            # else:
            #     logits *= 6 # reward is ~[-1.3*6, 1.8*6]
            if episode_success:
                print('Env Success episode, rewards:', [ep.reward for ep in episode_trans])
             
            for i in range(logits.shape[0]):  
                rew = logits[i, 0] # * 50 if self._reward_model.predict_logits else logits[i, 0] * 6
                episode_trans[i].reward = rew 
            if episode_success:
                print('Env Success episode, rewards after aug', [ep.reward for ep in episode_trans])
            
        for ep in episode_trans: 
            yield ep, episode_success