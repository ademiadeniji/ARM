Install on RLL servers:

New: for Slow resnets
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`

- `pip install wandb hydra pyrender einops MulticoreTSNE pandas bokeh fvcore matplotlib`
- `conda install jupyter`


```
ssh -T git@github.com

conda create -n arm python=3.7 pytorch==1.8.1
conda activate arm 
# If starting from scratch: 
# 1. install CoppeliaSim: 
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz 
tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz 
# NOTE version 4.1 and Ubuntu18_04!
# add to .bashrc:
export COPPELIASIM_ROOT=/home/mandi/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

source .bashrc
conda activate arm

# 2. VirtualGL rendering 
sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb --no-check-certificate 
sudo dpkg -i virtualgl*.deb
sudo X
nohup sudo X &
export DISPLAY=:0.0

# 3. PyRep 
git clone git@github.com:stepjam/PyRep.git
pip install -r PyRep/requirements.txt
pip install PyRep/.
# make sure example scripts work:
python PyRep/examples/example_baxter_pick_and_pass.py

# 4. RLBench -> YARR -> ARM 
git clone git@github.com:stepjam/RLBench.git
pip install -r RLBench/requirements.txt
pip install RLBench/.


git clone git@github.com:MandiZhao/YARR.git # (notice forked repo)
cd YARR; git checkout dev; cd ..
pip install -r YARR/requirements.txt
pip install -e YARR/.


git clone git@github.com:rll-research/ARM.git
pip install -r ARM/requirements.txt
# (Mandi) Added data/ and log/ folder, changed default settings in yaml
cp -r -n /shared/mandi/rlbench_demo/*  ARM/data/
# test launch
python launch.py method=C2FARM rlbench.task=stack_wine framework.gpu=0

```



```
sudo fuser -k /dev/nvidia3 
```
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
