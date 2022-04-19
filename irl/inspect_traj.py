import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from natsort import natsorted
from glob import glob
DATA_PATH=f'/home/mandi/ARM/irl/CLIP_ARM/push_button'
    
def main():
    data_itr = 2
    data_path = join(DATA_PATH, f'iteration{data_itr}')
    success_paths = natsorted(glob(f'{data_path}/success/episode*'))
    fail_paths = natsorted(glob(f'{data_path}/fail/episode*'))

    # for path in success_paths:
    for path in fail_paths: 
        print(path)
        episode = []
        steps = natsorted(glob(f'{path}/*.pkl'))
        #if 'success' in path or (len(steps) == 10 and 'fail' in path): # use this for iter 0
        for step in steps:
            with open(step, 'rb') as f:
                transition = pkl.load(f)
            print(transition.observation['front_rgb'].shape)
            raise ValueError
            episode.append(transition.observation['front_rgb'])
        n_steps = len(episode)
        fig, axs = plt.subplots(nrows=1, ncols=n_steps, squeeze=False, figsize=(n_steps*4, 4))
  
        for i in range(n_steps):
            #print(episode[i].shape, episode[i].transpose(1,2,0).shape)
            axs[0,i].imshow(episode[i].transpose(1,2,0))
            axs[0,i].axis('off')
        plt.savefig(f'{path}/vis.png')
 
    return


if __name__ == "__main__":
    main()