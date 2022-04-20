import clip 
import torch 
import pickle
import argparse
import numpy as np
from models import RewardMLP
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
from PIL import Image
from os.path import join

DATA_PATH='/home/mandi/ARM/irl/CLIP_ARM'

def main():
    parser = argparse.ArgumentParser(description='inspector')
    parser.add_argument('--path', type=str, default='/home/mandi/ARM/irl/clip-tuned/', help='path to model checkpoints')
    parser.add_argument('--model', type=str, default='demo_only_256x512', help='tuned model name')
    parser.add_argument('--step', type=str, default='250', help='tuned model saved step')
    
    parser.add_argument('--task', type=str, default='push_button', help='task name')
    parser.add_argument('--vis_demo', action='store_true', help='visualize demo')
    parser.add_argument('--traj_path', type=str, default='iteration0/fail/', help='traj data path')
    parser.add_argument('--episodes', type=int, default=5, help='num. of episodes')
    parser.add_argument('--vis_name', type=str, default='vis', help='vis name')
    args = parser.parse_args()


    device = 'cuda:0'
    model_name = "ViT-L/14" 
    clip_model, preprocess = clip.load(model_name, device=device)
    print('loaded CLIP model')
    prompts = ["robot gripper tip touch square button"]
    text = clip.tokenize(prompts).to(device)
    print('Using prompt', prompts)
    mlp = RewardMLP(clip_model=clip_model).to(device)
    mlp.load_state_dict(torch.load(f'{args.path}/{args.model}/{args.step}.pt'))
    print('Loaded model from {}'.format(args.model))
    
    # load data 
    if args.vis_demo:
        var = 0
        num_eps = min(int(args.episodes), len(glob(f'/home/mandi/all_rlbench_data/{args.task}/variation{var}/episodes/*')))
        paths = [
            f'/home/mandi/all_rlbench_data/{args.task}/variation{var}/episodes/episode{eps}/front_rgb/*' \
                for eps in range(num_eps)
        ]
        all_imgs = []
        for path in paths:
            frames = natsorted(glob(path)) 
            # get every nth frame:
            frames = [frames[i] for i in range(0, len(frames), 10)]
            imgs = [Image.open(p) for p in frames]
            all_imgs.append(imgs)
            
    else:
        num_eps = min(int(args.episodes), len(glob(join(DATA_PATH, args.task, args.traj_path, 'episode*'))))
        paths = [join(DATA_PATH, args.task, args.traj_path, f'episode{eps}') for eps in range(num_eps)]
        all_imgs = []
        for path in paths:
            steps = natsorted(glob(f'{path}/*.pkl'))
            imgs = []
            for i, step in enumerate(steps):
                with open(step, 'rb') as f:
                    transition = pickle.load(f)
                obs = transition.observation['front_rgb']
                if obs.shape[-1] != 3 and obs.shape[0] == 3:
                    obs = obs.transpose(1,2,0)
                obs = Image.fromarray(np.uint8(obs))
                imgs.append(obs)

                if i == len(steps) - 1:
                    final_obs = transition.final_observation['front_rgb']
                    if final_obs.shape[-1] != 3 and final_obs.shape[0] == 3:
                        final_obs = final_obs.transpose(1,2,0)
                        obs = Image.fromarray(np.uint8(final_obs))
                        imgs.append(obs)
            all_imgs.append(imgs)

    all_logits = []
    for imgs in all_imgs:
        img_batch = torch.stack([preprocess(img) for img in imgs]).to(device)
        logits = mlp(img_batch, text, scale_logits=False).detach().cpu().numpy()
        all_logits.append(logits)
    ymax, ymin = np.array(all_logits).max(), np.array(all_logits).min()
    
    max_traj_len = max([len(imgs) for imgs in all_imgs])
    fig, axs = plt.subplots(2 * len(all_imgs), max_traj_len, squeeze=False, figsize=(max_traj_len*4, 8*len(all_imgs)))
    for i, imgs in enumerate(all_imgs):
        for j, img in enumerate(imgs):
            r = 2*i
            axs[r,j].imshow(imgs[j])
            axs[r,j].axis('off')
            axs[r+1, j].bar(x=0, height=all_logits[i][j])
            axs[r+1, j].set_ylim(ymin, ymax)
        #axs[1,i].axis('off')
    plt.tight_layout()
    fname = f'vis_{args.model}_step{args.step}_{args.vis_name}.png'
    print('Saving to {}'.format(fname))
    plt.savefig(fname)



if __name__ == '__main__':
    main()