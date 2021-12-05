import os 
from os.path import join 
import hydra 
import numpy as np 
from omegaconf import DictConfig, OmegaConf
from pyrep.const import RenderMode
from rlbench import ObservationConfig, CameraConfig
from multiprocessing import cpu_count
from arm.demo_dataset import MultiTaskDemoSampler, RLBenchDemoDataset, collate_by_id, PyTorchIterableDemoDataset
from arm.models.slowfast  import TempResNet
from arm.models.vq_utils import Codebook
from arm.models.utils import make_optimizer
from arm.models.vis_utils import generate_tsne
from einops import rearrange, reduce, repeat, parse_shape
import logging
from functools import partial 
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import wandb 


def vq_hinge(model, input, vq_cfg):
    data = torch.stack([ d.get('front_rgb') for collate_id, d in input.items()] ) 
    assert len(data.shape) == 6, 'Must be shape (b, n, num_frames, channels, img_h, img_w) '
    b, k, n, ch, img_h, img_w = data.shape
    device = model.get_device()
    model_inp = rearrange(data, 'b k n ch h w -> (b k) ch n h w').to(device)
    embeddings = model(model_inp) 
    if vq_cfg.latent_dim == 1:
        embeddings = rearrange(embeddings, '(b k) d -> b k d 1', b=b, k=k)
    elif vq_cfg.latent_dim == 3:
        embeddings = rearrange(model._conv_out, '(b k) d t h w -> b k d (t h w)', b=b, k=k)
     
    embeddings_norm = embeddings / embeddings.norm(dim=2, p=2, keepdim=True)


def hinge_loss(model, input, hinge_cfg):
     
    data = torch.stack([ d.get('front_rgb') for collate_id, d in input.items()] ) 
    assert len(data.shape) == 6, 'Must be shape (b, n, num_frames, channels, img_h, img_w) '
    b, k, n, ch, img_h, img_w = data.shape
    device = model.get_device()
    model_inp = rearrange(data, 'b k n ch h w -> (b k) ch n h w').to(device)
    embeddings = model(model_inp)
    
    embeddings = rearrange(embeddings, '(b k) d -> b k d', b=b, k=k)
     
    embeddings_norm = embeddings / embeddings.norm(dim=2, p=2, keepdim=True)

    support_embeddings = embeddings_norm[:, :hinge_cfg.num_support]
    query_embeddings = embeddings_norm[:, -hinge_cfg.num_query:].reshape(
        b * hinge_cfg.num_query, -1)

    support_context = support_embeddings.mean(1)  # (B, E)
    support_context = support_context / support_context.norm(
        dim=1, p=2, keepdim=True)
    similarities = support_context.matmul(query_embeddings.transpose(0, 1))
    similarities = similarities.view(b, b, hinge_cfg.num_query)  # (B, B, queries)

    # Gets the diagonal to give (batch, query)
    diag = torch.eye(b, device=device) 
    positives = torch.masked_select(similarities, diag.unsqueeze(-1).bool())  # (B * query)
    positives = positives.view(b, 1, hinge_cfg.num_query)  # (B, 1, query)

    negatives = torch.masked_select(similarities, diag.unsqueeze(-1) == 0)
    # (batch, batch-1, query)
    negatives = negatives.view(b, b - 1, -1)

    loss = torch.max(
        torch.zeros_like(negatives).to(device),  
        hinge_cfg.margin - positives + negatives)
    loss = loss.mean() * hinge_cfg.emb_lambda

    # Summaries
    max_of_negs = negatives.max(1)[0]  # (batch, query)
    accuracy = positives[:, 0] > max_of_negs
    embedding_accuracy = accuracy.float().mean()

    return {
        'context': support_context,
        'loss': loss,
        'embed_accuracy': embedding_accuracy,
    }

def make_dataset(cfg, mode, obs_config):
    assert mode in ['train', 'val'], f'Got unexpected mode {mode}'
    dataset = RLBenchDemoDataset(
        obs_config=obs_config,
        mode=mode,
        **cfg.dataset) 
    dataset = PyTorchIterableDemoDataset(
            demo_dataset=dataset,
            batch_dim=cfg.sampler.batch_dim if mode == 'train' else cfg.val_sampler.batch_dim,
            samples_per_variation=cfg.sampler.k_dim if mode == 'train' else cfg.val_sampler.k_dim,
            sample_mode=mode, 
            )

    return dataset

def make_loader(cfg, mode, dataset):
    variation_idxs, task_idxs = dataset.get_idxs()
    sampler = MultiTaskDemoSampler(
        variation_idxs_list=variation_idxs, # 1-1 maps from each variation to idx in dataset e.g. [[0,1], [2,3], [4,5]] belongs to 3 variations but 2 tasks
        task_idxs_list=task_idxs,      # 1-1 maps from each task to all its variations e.g. [[0,1,2,3], [4,5]] collects variations by task
        **(cfg.val_sampler if mode == 'val' else cfg.sampler),
    )

    collate_func = partial(collate_by_id, cfg.sampler.sample_mode+'_id')
    loader = DataLoader(
            dataset, 
            batch_sampler=sampler,
            num_workers=min(11, cpu_count()),
            worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w),
            collate_fn=collate_func,
            )
    return loader 
 
def validate(model, loss_fn, loss_cfg, step, cfg, dataset):
    # make a new val loader 
    # val_loader = make_loader(cfg, 'val', dataset)

    model = model.eval() 
    mean_val_loss, val_count = 0, 0
    mean_acc = 0
    model = model.eval()
    for _ in cfg.train.val_steps:
        val_inp = next(dataset) # assume iterable 
        loss_out = loss_fn(model, val_inp, loss_cfg)
        mean_val_loss += loss_out['loss'].item()
        mean_acc += loss_out['embed_accuracy']
        val_count += 1
    wandb.log(
        {   'Val At Step': i, 
            'Val Loss': (mean_val_loss / val_count),
            'Val Embed Accuracy': (mean_acc / val_count)
    }  
    )


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
    train_dataset, val_dataset = [
        make_dataset(cfg, mode, obs_config) for mode in ['train', 'val']]
    
    
 
    model = TempResNet(cfg.encoder)
    device = torch.device('cuda:0') if not cfg.train.use_cpu else torch.device('cpu')
    model.set_device(device)
    model = model.train()

    # generate_tsne(model, val_dataset)
    # raise ValueError
     
    optimizer = make_optimizer(model, cfg=cfg.encoder)

    if cfg.train.loss_type == "hinge":
        loss_fn = hinge_loss
        loss_cfg = cfg.hinge_cfg
    elif cfg.train.loss_type == "vq-hinge":
        loss_fn = vq_hinge 
        loss_cfg = cfg.vq_cfg 
        model.codebook = Codebook(
            n_codes=loss_cfg.n_codes,
            embedding_size=loss_cfg.embedding_size)
    else:
        raise NotImplementedError

    cwd = os.getcwd()
    cfg.train.run_name = "-".join([
        f"{cfg.train.run_name}",
        f"batch{cfg.sampler.batch_dim}x{cfg.sampler.k_dim}",
        f"lr{cfg.encoder.OPTIM.BASE_LR}",
        f"hinge{cfg.hinge_cfg.num_support}x{cfg.hinge_cfg.num_query}"
        ])
    log_path = join(cwd, cfg.train.run_name)
    cfg.train.log_path = log_path
    os.makedirs(log_path, exist_ok=('burn' in log_path or cfg.train.overwrite))
    weightsdir = join(log_path, 'weights')
    print(f'Saving weights to dir: {weightsdir}')
    os.makedirs(weightsdir, exist_ok=('burn' in log_path or cfg.train.overwrite))
    tsne_dir = join(log_path, 'tsnes')
    os.makedirs(tsne_dir, exist_ok=('burn' in log_path or cfg.train.overwrite))
    OmegaConf.save( config=cfg, f=join(log_path, 'config.yaml') )
    
    print('Initializing wandb run with keywords:', cfg.wandb)
    run = wandb.init(project='MTARM', **cfg.wandb)
    run.name = cfg.train.log_path.split('/')[-1]
    cfg_dict = {}
    for key in cfg.keys():
        if key == 'defaults':
            for sub_key, val in cfg[key]['encoder'].items():
                cfg_dict['encoder/'+sub_key] = val 
        else:
            for sub_key in cfg[key].keys():
                cfg_dict[key+'/'+sub_key] = cfg[key][sub_key]
    run.config.update(cfg_dict)
    run.save()
 
 
    train_iter = iter(train_dataset)
    val_iter = iter(val_dataset)
    for i in range(cfg.train.steps):
        # the custom multi-task sampler doesn't refresh properly, hack here
        # train_loader = make_loader(cfg, 'train', train_dataset)
        # val_loader  = make_loader(cfg, 'val', val_dataset)  
        model = model.train()
        if i % cfg.train.val_freq == 0:
            validate(model, loss_fn, loss_cfg, i, cfg, val_iter)
        
        inputs = next(train_iter)
        optimizer.zero_grad()
        loss_out = loss_fn(model, inputs, loss_cfg)
        loss = loss_out['loss']
        loss.backward()
        optimizer.step()
            
        if step % cfg.train.log_freq == 0:
            wandb.log({   
                'Train Step': i, 
                'Train Loss': loss.item(),
                'Train Embed Accuracy': loss_out['embed_accuracy'],
                })

        if i % cfg.train.save_freq == 0:
            savedir = join(weightsdir, str(i))
            os.makedirs(savedir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(savedir, 'checkpoint.pt'))
        
        if i % cfg.train.vis_freq == 0:
            # gen html file and log to wandb 
            generate_tsne(
                model, 
                i, 
                val_dataset, 
                log_dir=tsne_dir,
                num_task=-1, 
                num_vars=-1, 
                num_img_frames=min(5, cfg.dataset.num_steps_per_episode))


    
if __name__ == '__main__':
    main()
