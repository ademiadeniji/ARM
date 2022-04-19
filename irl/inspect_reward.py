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
    parser.add_argument('--traj_path', type=str, default='iteration1/success/', help='traj data path')
    parser.add_argument('--episode', type=int, default=1, help='episode number')
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
        eps = int(args.episode)
        path = f'/home/mandi/all_rlbench_data/{args.task}/variation{var}/episodes/episode{eps}/front_rgb/*'
        frames = natsorted(glob(path)) 
        # get every nth frame:
        frames = [frames[i] for i in range(0, len(frames), 10)]
        imgs = [Image.open(p) for p in frames]
       
    else:
        path = join(DATA_PATH, args.task, args.traj_path, f'episode{args.episode}') 
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

    img_batch = torch.stack([preprocess(img) for img in imgs]).to(device)
    logits = mlp(img_batch, text, scale_logits=False).detach().cpu().numpy()
    ymax, ymin = logits.max(), logits.min()
    

    fig, axs = plt.subplots(2, len(imgs), squeeze=False, figsize=(len(imgs)*4, 8))
    for i in range(len(imgs)):
        axs[0,i].imshow(imgs[i])
        axs[0,i].axis('off')
        axs[1,i].bar(x=0, height=logits[i])
        axs[1, i].set_ylim(ymin, ymax)
        #axs[1,i].axis('off')
    plt.tight_layout()
    plt.savefig(f'vis_{args.model}_step{args.step}_{args.vis_name}.png')



if __name__ == '__main__':
    main()