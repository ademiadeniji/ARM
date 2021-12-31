# dVAE all tasks

DIM=16
HID=1024 
RUN=1frame-dVAE-QEncode${DIM}-Hidden${HID}-UpdatePrio-Switch1 
taskset -c $CPUS  python launch_context.py run_name=$RUN tasks='${10_tasks}'   \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=100000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=1 \
replay.buffers_per_batch=12 framework.log_freq=500 framework.switch_online_tasks=1 \
method.voxel_sizes=[16,16,16] method.bounds_offset=[0.15,0.15]

DIM=16 
RUN=10Var-OneHot-QEncode${DIM} 
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN tasks=['pick_and_lift']   \
dev.one_hot=True  dev.qnet_context_latent_size=${DIM} framework.training_iterations=50000 \
dev.single_layer_context=True 

# debug cup
DIM=16
HID=1024
OUT=512
RUN=10Var-DEBUG-3frameConv3d-dVAE-QEncode${DIM}-Hidden${HID}-Ker311-Feat${OUT}-Stride2
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN   \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=3 cdev.use_conv=True cdev.conv_out=${OUT} cdev.stride=2 dev.ctxt_conv3d=True \
framework.wandb=False rlbench.demos=1 dev.offline=True 

DIM=16
HID=1024
OUT=4096
RUN=10Var-3frameConv3d-dVAE-QEncode${DIM}-Hidden${HID}-Ker311-Feat${OUT}-Stride1-RERUN
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN tasks=['pick_and_lift'] \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=3 cdev.use_conv=True cdev.conv_out=${OUT} cdev.conv_kernel=[3,1,1]

DIM=16
HID=1024 
RUN=10Var-5frameStack-dVAE-QEncode${DIM}-Hidden${HID} 
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN tasks=['pick_and_lift'] \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=5

RUN=10Var-3Cam-OneHot
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN tasks=['pick_and_lift'] \
framework.training_iterations=20000  dev.one_hot=True \
rlbench.cameras=['front','left_shoulder','right_shoulder']

RUN=3Cam-OneHot
taskset -c $CPUS  python launch_context.py rlbench.num_vars=3 run_name=$RUN tasks=['put_groceries_in_cupboard'] \
framework.training_iterations=20000  dev.one_hot=True  \
rlbench.cameras=['front','left_shoulder','right_shoulder']

DIM=16
HID=1024
RUN=2Var-3Cam-1frame-dVAE
taskset -c $CPUS  python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=['take_money_out_safe'] \
rlbench.cameras=['front','left_shoulder','right_shoulder'] \
replay.batch_size=30 replay.buffers_per_batch=2 \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=1

# multi-tcam (left_shoulder + right_shoulder): meat_griff, put_procery, money, unplug_charger, 
# put_groceries_in_cupboard: 9 vars
# meat_off_grill and meat_on_grill: 2 vars
# put_money_in_safe, take_money_out_safe: 3 each
# unplug_charger: 1 var

RUN=2Var-3Cam-OneHot
taskset -c $CPUS  python launch_context.py rlbench.num_vars=-1 run_name=$RUN \
tasks=['meat_off_grill'] framework.training_iterations=20000  dev.one_hot=True \
replay.batch_size=30 replay.buffers_per_batch=2 \
rlbench.cameras=['front','left_shoulder','right_shoulder']

# mt-10, one-hot
RUN=1Cam-OneHot-10TaskBatch-Ratio12
taskset -c $CPUS  python launch_context.py rlbench.num_vars=-1 run_name=$RUN \
tasks.heldout='pick_up_cup' dev.one_hot=True  framework.replay_ratio=12 \
replay.batch_size=6 replay.buffers_per_batch=25 replay.num_tasks_per_batch=10  \
replay.replay_size=1000000 framework.training_iterations=100000 framework.log_freq=500

# single task
RUN=6Var-Wrist4Cam-OneHot
taskset -c $CPUS  python launch_context.py rlbench.num_vars=-1 run_name=$RUN \
dev.one_hot=True tasks='Single_1cam' tasks.all_tasks=['put_groceries_in_cupboard'] tasks.num_vars=6 \
rlbench.cameras=['front','left_shoulder','right_shoulder','wrist']
