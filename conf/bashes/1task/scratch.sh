# 1 buffer case:
Task=stack_wine 
DIM=16
HID=2048 
RATIO=36
for DEMO in 3 4
do 
    RUN=Scratch-1Cam-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
    taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${Task}] \
    framework.replay_ratio=$RATIO tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True \
    contexts.loss_mode='dvae' dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
    rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=60 replay.buffers_per_batch=1 \
    tasks.train_steps=10000 framework.num_log_episodes=10 framework.eval_envs=2 replay.replay_size=100000
done


# pick up cup has more vars!
Task=[reach_target]
DIM=16
HID=2048 
RATIO=30
DEMO=1
for DEMO in 10
do 
    RUN=Scratch-1Cam-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
    taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=${Task} \
    framework.replay_ratio=$RATIO tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True \
    contexts.loss_mode='dvae' dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
    rlbench.demos=${DEMO} framework.ckpt_eval=True \
    tasks.train_steps=10000 framework.num_log_episodes=10 framework.eval_envs=2 \
    replay.batch_size=6 replay.buffers_per_batch=10 
done


# debug with push_button 
Task=[push_button]
DIM=16
HID=2048 
RATIO=12 
DEMO=0

for TRANS in 100 200 10
do
RUN=Iterate-${TRANS}trans 
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=${Task} \
framework.replay_ratio=$RATIO tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True \
contexts.loss_mode='dvae' dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
rlbench.demos=${DEMO} framework.ckpt_eval=True tasks.train_steps=1000 framework.eval_envs=2 \
framework.num_log_episodes=5 framework.transitions_before_train=${TRANS}

done 


# try single buffer 
Task=[pick_up_cup]
RATIO=6
RUN=OneBuffer-OneHot-${RATIO}
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=${Task}  \
dev.one_hot=True replay.batch_size=60 replay.buffers_per_batch=1 \
framework.replay_ratio=$RATIO replay.share_across_tasks=True \
rlbench.demos=3 tasks.train_steps=10000

# 1var for pick up
Task=pick_up_cup 
DIM=16
HID=2048 
RATIO=36
for DEMO in 0 1
do 
    RUN=Scratch-1Var-1Cam-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
    taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${Task}] \
    framework.replay_ratio=$RATIO tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True \
    contexts.loss_mode='dvae' dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
    rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=60 replay.buffers_per_batch=1 \
    tasks.train_steps=10000 framework.num_log_episodes=10 framework.eval_envs=2 replay.replay_size=100000 \
    rlbench.num_vars=1
done


# no ckpt eval
RUN=Scratch-AugBatch6-1Cam-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=${Task} \
framework.replay_ratio=$RATIO tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True \
contexts.loss_mode='dvae' dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
rlbench.demos=${DEMO}  \
tasks.train_steps=10000  \
replay.batch_size=6 replay.buffers_per_batch=10 dev.augment_batch=6 