import os 
from os.path import join 
import hydra 
import numpy as np 
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from pyrep.const import RenderMode
from rlbench import ObservationConfig, CameraConfig
from multiprocessing import cpu_count
from arm.demo_dataset import MultiTaskDemoSampler, RLBenchDemoDataset, collate_by_id
from arm.models.slowfast  import TempResNet
from arm.models.utils import make_optimizer
from einops import rearrange, reduce, repeat, parse_shape
import logging
from functools import partial 
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import wandb
from glob import glob 
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10, cividis, turbo 
import pandas as pd
from io import BytesIO
import base64
import PIL
from PIL import Image 
 
# for bokeh
WIDTH=1000 
HEIGHT=800 
NUM_VIS=3
NUM_TASKS=100
NUM_VARS=5 
DOT_SIZE=8
HOV_H=100
HOV_W=100
def encode_images(images):
    """ bokeh requires this weird encoding thing to display hover-over images """
    encodes = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format='png')
        for_encoding = buffer.getvalue()
        encoded = 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
        encodes.append(encoded)
    return encodes

def generate_tsne(
    model, 
    step, 
    dataset, 
    log_dir='/home/mandi/ARM/', 
    num_task=-1, 
    num_vars=-1, 
    num_img_frames=1):
    # when visualize during embedding training run,
    # dataset should be the same the model gets train/validated on
    model = model.eval()
    embeds = []
    names = []
    data_list = dataset.get_some_data(num_task, num_vars)
    hovers = []
    
    for data in data_list: 
        inp = data['front_rgb']
        
        n, ch, img_h, img_w = inp.shape
        device = model.get_device()
        inp = rearrange(inp, 'n ch h w -> 1 ch n h w').to(device)

        embeds.append( model( inp ).detach().cpu().numpy()[0] )
        names.append( data['name'] ) 
    
        num_frames = min(len(data['all_imgs']), num_img_frames)
        array_seq = np.concatenate( [np.array(ob) for ob in data['all_imgs'][-num_frames:] ], axis=1) # each img is 128,128,3
        img_seq = Image.fromarray(array_seq)
        hovers.append(img_seq)
        # single img: hovers.append(data['last_img']) # this is PIL.Image
    
    encoded_hovers = encode_images(hovers)

    print(f'TSNE step {step}: Generated embeddings for {len(embeds)} tasks, embedding shape {embeds[0].shape}')
    tsne = TSNE(n_jobs=4, )
    X = np.stack(embeds)
    Y = tsne.fit_transform(X)
 
    # html part:
    filename = join(log_dir, f'tsne_{step}.html')
    output_file( filename )
    plot_figure = figure(
        title=f'TSNE Clusters from Model Trained Step {step}',
        plot_width=WIDTH,
        plot_height=HEIGHT,
        tools=('pan, wheel_zoom, reset'),
        )
    plot_figure.add_tools(
        HoverTool(tooltips=f"""
        <div>
            <div>
                <img 
                    src='@image'  height="{HOV_H}" width="{HOV_W * num_frames}" 
                    style='float: left; margin: 0px 0px 0px 0px;'
                ></img>
            </div>

        </div>
        """))
    task_to_row = defaultdict(list)
    for row in range(Y.shape[0]):
        task_to_row[ data_list[row]['task_id'] ].append(row)
    
    for task_id, rows in task_to_row.items():
        datasource = ColumnDataSource({
        'x': list(Y[rows, 0]),
        'y': list(Y[rows, 1]), 
        'dw': [10 for r in rows],
        'dh': [10 for r in rows],
        'image': [encoded_hovers[r] for r in rows],
            })
        plot_figure.circle(
            'x',
            'y',
            color=turbo(NUM_TASKS)[task_id],
            legend_label=f'task_{task_id}',
            line_alpha=0.9,
            fill_alpha=0.9,
            size=DOT_SIZE,
            source=datasource,
        )
 
    plot_figure.legend.click_policy="mute"
 
    show(plot_figure)
    wandb.log(
            {   'TSNE': wandb.Html(filename),
                'TSNE Model Step': step,
        }  
        )

    return 
