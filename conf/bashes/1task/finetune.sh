# rtxl2 
TASK=take_lid_off_saucepan
DIM=16
HID=2048
DEMO=0
for RATIO in 36  
do
RUN=FineTune-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
RES=10Task-84var-Heldout-take_lid_off_saucepan/1Cam-dVAE-3frameStack-10TaskBatch-Ratio24-gpu012-Replay_B6x25-Hidden2048-Encode16/seed0 
STEP=20500 
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${TASK}] \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=60 replay.buffers_per_batch=1 \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO}  \
 tasks.train_steps=10000 framework.num_log_episodes=10 framework.eval_envs=2 

done 

TASK=put_rubbish_in_bin
DIM=16
HID=2048
RES=10Task-84var-Heldout-put_rubbish_in_bin/1Cam-dVAE-3frameStack-10TaskBatch-Ratio24-Replay_B6x15-Hidden2048-Encode16/seed0
RATIO=60 
RUN=FineTune-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
STEP=11000 
DEMO=0 
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${TASK}] \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=60 replay.buffers_per_batch=1 \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO}

# rtxl1 
TASK=stack_wine
DIM=16
HID=2048
RES=10Task-84var-Heldout-stack_wine/1Cam-dVAE-3frameStack-10TaskBatch-RandomRest-Ratio36-Replay_B6x35-Hidden2048-Encode16/seed0 
RATIO=90
STEP=18600 
DEMO=1
RUN=FineTune-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${TASK}] \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=60 replay.buffers_per_batch=1 \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO} \
 framework.num_log_episodes=10 framework.eval_envs=2

# rtxl1
TASK=pick_up_cup
DIM=16
HID=2048 
RATIO=60
RES=10Task-65var-Heldout-pick_up_cup/1Cam-dVAE-3frameStack-10TaskBatch-Ratio12-Replay_B6x25-Hidden1024-Encode16/seed0
STEP=15600 
DEMO=0 
RUN=FineTune-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${TASK}] \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=6 replay.buffers_per_batch=10 \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO} \
  tasks.train_steps=10000 framework.num_log_episodes=10 framework.eval_envs=2 

# pabti5 
TASK=push_button
RES=10Task-67var-Heldout-push_button/1Cam-dVAE-3frameStack-10TaskBatch-Ratio24-Replay_B6x20-Hidden1024-Encode16/seed0
STEP=22000
DIM=16
HID=2048 
RATIO=6
DEMO=0 
RUN=FineTune-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${TASK}] \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=6 replay.buffers_per_batch=10 \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO} \
  tasks.train_steps=10000 framework.num_log_episodes=10 framework.eval_envs=2 


# zero-shot eval a checkpoint 
RES=
STEP= 
DIM=16
HID=2048 
RATIO=0
DEMO=0 
TASK=push_button
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${TASK}] \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=3 replay.buffers_per_batch=3 \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO} \
  tasks.train_steps=10000 framework.num_log_episodes=10 framework.train_envs=0 framework.wandb=False framework.transitions_before_train=0