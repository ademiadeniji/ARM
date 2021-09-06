# Attention-driven Robotic Manipulation (ARM)

Codebase of Q-attention (within the ARM system) and coarse-to-fine Q-attention (within C2F-ARM system) from the following papers:

- [Q-attention: Enabling Efficient Learning for Vision-based Robotic Manipulation](https://arxiv.org/abs/2105.14829) (ARM system)
- [Coarse-to-Fine Q-attention: Efficient Learning for Visual Robotic Manipulation via Discretisation](https://arxiv.org/abs/2106.12534) (C2F-ARM system)

![task grid image missing](readme_files/arm_c2farm.png)

Not that C2F-ARM is the better performing system.

## Installation

ARM is trained using the **YARR framework**. Head to the [YARR github](https://github.com/stepjam/YARR) page and follow 
installation instructions.

ARM is evaluated on **RLBench** 1.1.0. Head to the [RLBench github](https://github.com/stepjam/RLBench) page and follow 
installation instructions. 

Now install project requirements:
```bash
pip install -r requirements.txt
```

New: for Slow resnets
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`

- `pip install einops MulticoreTSNE pandas bokeh fvcore matplotlib`
- `conda install jupyter`


## Running experiments

Be sure to have RLBench demos saved on your machine before proceeding. To generate demos for a task, go to the 
tools directory in RLBench _(rlbench/tools)_, and run:
```bash
DATAP=/home/mandi/front_rlbench_data
python dataset_generator.py --save_path=$DATAP --tasks=lamp_on --image_size=128,128 \
--renderer=opengl --episodes_per_task=20 --variations=-1 --processes=-1


TASKS=
python ../RLBench/tools/dataset_frontcam_gen.py --save_path=$DATAP --image_size=128,128 \
--renderer=opengl --episodes_per_task=20 --variations=-1 --processes=20

```


Experiments are launched via [Hydra](https://hydra.cc/). To start training C2F-ARM on the 
**take_lid_off_saucepan** task with the default parameters on **gpu 0**:
```bash
python launch.py method=C2FARM rlbench.task=take_lid_off_saucepan rlbench.demo_path=/mnt/my/save/dir framework.gpu=0
```
