import numpy as np
from glob import glob 
from natsort import natsorted
import matplotlib.pyplot as plt
from PIL import Image
import os 
## one cam, all episodes
task = 'take_money_out_safe'
fname = 'sample_episodes'
n_frames = 3
cam_name = 'right_shoulder'
for var_id in range(1, 3):
    episodes = natsorted(glob(f'/home/mandi/all_rlbench_data/{task}/variation{var_id}/episodes/*'))
    fig, axs = plt.subplots(ncols=n_frames+1, nrows=len(episodes), figsize=((n_frames+1)*4, len(episodes)*4))
    for row, p in enumerate(episodes):
        imgs = natsorted(glob(p + f'/{cam_name}_rgb/*'))
        tots = len(imgs)
        arrs = np.array_split(np.arange(tots), n_frames)
        idxs = [int(np.random.choice(ls,1)) for ls in arrs]
        idxs.append(-1)

        print('sampling:', imgs[-1])
        for col, idx in enumerate(idxs):
            axs[row, col].imshow(Image.open(imgs[idx]))
            axs[row, col].axis('off')

    plt.tight_layout()
    #plt.savefig(f'/home/mandi/ARM/{task}_var{var_id}_{n_frames}frames.png')
    os.makedirs(f'/home/mandi/ARM/pngs/{task}/{fname}', exist_ok=True)
    plt.savefig(f'/home/mandi/ARM/pngs/{task}/{fname}/var{var_id}_{cam_name}_{n_frames}frames.png')

## all cam, one episode
tasks = [
    'pick_and_lift',
    'unplug_charger',
    'pick_up_cup',
    'phone_on_base',
    
    'put_rubbish_in_bin',
    'reach_target',
    'stack_wine', 
    'take_lid_off_saucepan',
    'take_umbrella_out_of_umbrella_stand',
    'lamp_on',
    'lamp_off',
    'open_door',
    'press_switch',
    'push_button',
    'take_usb_out_of_computer',
    'close_drawer',
    'meat_off_grill', 
    'put_groceries_in_cupboard', 
    'put_money_in_safe', 
    
]
tasks = ['take_money_out_safe']

# n_frames = 5
# var_id = 0
# fname = 'sample_cams'

# for task in tasks:
#     cams = [g for g in \
#         glob(f'/home/mandi/all_rlbench_data/{task}/variation{var_id}/episodes/episode0/*') if 'rgb' in g]
#     fig, axs = plt.subplots(ncols=n_frames, nrows=len(cams), figsize=(n_frames*4, len(cams)*4))
#     for row, p in enumerate(cams):
#         imgs = natsorted(glob(p + '/*'))
#         cam_name = p.split('/')[-1]
#         arrs = np.array_split(np.arange(len(imgs)), n_frames)
#         idxs = [int(np.random.choice(ls,1)) for ls in arrs]
#         print('sampling:', imgs[-1])
#         for col, idx in enumerate(idxs):
#             ax = axs[row, col]
#             ax.imshow(Image.open(imgs[idx]))
#             ax.axis('off')
#             ax.set_title(cam_name, fontsize=15)
    
#     os.makedirs(f'/home/mandi/ARM/pngs/{task}/{fname}', exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(f'/home/mandi/ARM/pngs/{task}/{fname}/var{var_id}_{n_frames}frames.png')


