# see of "readable" embeddings can be useful for new multi-var task! 
RES=pick_up_cup-20var/Scratch-1Cam-dVAE-3frameStack-Ratio12-Demo1-Replay_B6x10/seed1
STEP=9999
DIM=16
HID=2048 
DEMO=0 
TASK=pick_and_lift
RATIO=30 
ASIZE=6
RUN=1Task-AugBatch${ASIZE}-FineTune-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${TASK}] \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=6 replay.buffers_per_batch=10 \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO} \
  tasks.train_steps=20000 framework.num_log_episodes=5 framework.eval_envs=4 \
  dev.augment_batch=${ASIZE} resume_path=/shared/mandi/arm_log 

RUN=1Task-FineTune-Freeze2Layer-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${TASK}] \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=6 replay.buffers_per_batch=10 \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO} \
  tasks.train_steps=20000 framework.num_log_episodes=5 framework.eval_envs=4 \
  resume_path=/shared/mandi/arm_log resume_freeze=[0,1]