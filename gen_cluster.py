"""
Use this to load back checkpoints and make TSNE
"""
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

def make_dataset(cfg, mode, obs_config):
    assert mode in ['train', 'val'], f'Got unexpected mode {mode}'
    dataset = RLBenchDemoDataset(
        obs_config=obs_config,
        mode=mode,
        **cfg.dataset) 
    return dataset 

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

def generate_tsne_v2(model, dataset, step, num_tasks=100, variation_per_task=1, filename='/home/mandi/ARM/tsne_step0'):
    model = model.eval()
    embeds = []
    names = []
    data_list = dataset.get_some_data(num_tasks, variation_per_task)
    hovers = []
    
    for data in data_list: 
        inp = data['front_rgb']
        
        n, ch, img_h, img_w = inp.shape
        device = model.get_device()
        inp = rearrange(inp, 'n ch h w -> 1 ch n h w').to(device)

        embeds.append( model( inp ).detach().cpu().numpy()[0] )
        names.append( data['name'] ) 
        hovers.append(data['last_img']) # this is PIL.Image
    
    encoded_hovers = encode_images(hovers)

    print(f'Embedded {len(embeds)} tasks, embedding shape {embeds[0].shape}')
    tsne = TSNE(n_jobs=4, )
    X = np.stack(embeds)
    Y = tsne.fit_transform(X)
    print(f'TSNE fitted shape: {Y.shape}')
    
    # html part:
    output_file(f'{filename}.html')
    plot_figure = figure(
        title=f'TSNE Clusters from Model Trained Step {step}',
        plot_width=WIDTH,
        plot_height=HEIGHT,
        tools=('pan, wheel_zoom, reset'),
        )
    plot_figure.add_tools(
        HoverTool(tooltips="""
        <div>
            <div>
                <img 
                    src='@image'  height="62" width="62" 
                    style='float: left; margin: 0px 0px 0px 0px;'
                ></img>
            </div>

        </div>
        """))
    task_to_row = defaultdict(list)
    for row in range(Y.shape[0]):
        task_to_row[
            data_list[row]['task_id']
            ].append(row)
    
    
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
        # plot_figure.image(
        # source=datasource, 
        # image='image', 
        # x='x', 
        # y='y',  )
    plot_figure.legend.click_policy="mute"
 
    show(plot_figure)

    return 


def generate_tsne_html(model, dataset, step, num_tasks=100, variation_per_task=1, filename='/home/mandi/ARM/tsne_step0'):
    model = model.eval()
    embeds = []
    names = []
    data_list = dataset.get_some_data(num_tasks, variation_per_task)
    hovers = []
    for data in data_list: 
        inp = data['front_rgb']
        
        n, ch, img_h, img_w = inp.shape
        device = model.get_device()
        inp = rearrange(inp, 'n ch h w -> 1 ch n h w').to(device)

        embeds.append( model( inp ).detach().cpu().numpy()[0] )
        names.append( data['name'] ) 
        hovers.append(data['last_img']) # this is PIL.Image
    
    encoded_hovers = encode_images(hovers)

    print(f'Embedded {len(embeds)} tasks, embedding shape {embeds[0].shape}')
    tsne = TSNE(n_jobs=4, )
    X = np.stack(embeds)
    Y = tsne.fit_transform(X)
    print(f'TSNE fitted shape: {Y.shape}')
    
    # html part:
    output_file(f'{filename}.html')
    plot_figure = figure(
        title=f'TSNE Clusters from Model Trained Step {step}',
        plot_width=WIDTH,
        plot_height=HEIGHT,
        tools=('pan, wheel_zoom, reset'),
        )
 
    color_mapping = CategoricalColorMapper(
        factors=[ str(i) for i in range(NUM_TASKS)],
        palette=turbo(NUM_TASKS) )
                                       
    datasource = ColumnDataSource({
        'x': list(Y[:, 0]),
        'y': list(Y[:, 1]),
        'image': encoded_hovers,
        'dw': [10 for _ in encoded_hovers],
        'dh': [10 for _ in encoded_hovers],
        'task_id': [ str(d['task_id']) for d in data_list],
        'label': [ 'task{}'.format(d['task_id']) for d in data_list],
        'color': [ turbo(NUM_TASKS)[d['task_id']] for d in data_list]

    })

    plot_figure.add_tools(
        HoverTool(tooltips="""
        <div>
            <div>
                <img 
                    src='@image'  height="62" width="62" 
                    style='float: left; margin: 0px 0px 0px 0px;'
                ></img>
            </div>

        </div>
        """))
    

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        #color=dict(field='task_id', transform=color_mapping),
        color='color',
        legend_field='label',
        line_alpha=0.9,
        fill_alpha=0.9,
        size=DOT_SIZE,
        
    )
    plot_figure.legend.click_policy="mute"
    # plot_figure.image(
    #     source=datasource, 
    #     image='image', 
    #     x='x', 
    #     y='y', 
    #     dw='dw', 
    #     dh='dh',)

    show(plot_figure)

    return 


@hydra.main(config_name='context_cfg', config_path='conf')
def main(cfg: DictConfig) -> None: 

    one_cam = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=cfg.dataset.image_size,
        render_mode=RenderMode.OPENGL)

    obs_config = ObservationConfig(
        front_camera=one_cam)
    val_dataset = make_dataset(cfg, 'val', obs_config)  
    
    load_dir = cfg.train.log_path
    assert os.path.exists(load_dir), f'Tring to load from {cfg.train.log_path} but not found'
    cfg = OmegaConf.load(
        join(load_dir, 'config.yaml'),)
    model = TempResNet(cfg.encoder)
    device = torch.device('cuda:0') if not cfg.train.use_cpu else torch.device('cpu')
    model.set_device(device)
    model = model.eval()

    cfg.wandb.job_type = 'visualize'  # modifying the loaded dir 
    cfg.wandb.group = load_dir
    run = wandb.init(project='MTARM', **cfg.wandb)

    checkpoints = sorted(glob( join(load_dir, 'weights/*/checkpoint.pt')))
    assert len(checkpoints) > 0, f"Found no checkpoints for {load_dir}"
    
    msteps = sorted(
        [ int(model_path.split('/')[-2]) for model_path in checkpoints])
    take_steps = [np.min(msteps)] + \
        list(np.random.choice(msteps[1:-1], NUM_VIS, replace=False)) + \
         [np.max(msteps)]
    for step in take_steps:
        model_path = join(load_dir, 'weights', str(step), 'checkpoint.pt')
         
        model.load_state_dict(
                    torch.load(
                        model_path,
                        map_location=model.get_device())
                    )
        fname = join('/home/mandi/ARM/tsne_step{}'.format(step))
        print(f'Generating tsne and save html to: {fname}')
        generate_tsne_v2(
            model, 
            val_dataset, 
            step,
            num_tasks=NUM_TASKS, 
            variation_per_task=NUM_VARS, 
            filename=fname)
        wandb.log(
            {   'TSNE': wandb.Html(fname+'.html'),
                'Model Step': step,
        }  
        )
 
    
     


if __name__ == '__main__':
    main()
