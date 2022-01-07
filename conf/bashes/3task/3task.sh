# try different batch arrangement with 1 multi-variation task and 2 single-variation tasks

DIM=16
HID=2048
RATIO=6
RUN=1Cam-1frame-EqualVal 
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID}  \
replay.batch_size=6 replay.buffers_per_batch=15 \
replay.num_tasks_per_batch=3 framework.replay_ratio=$RATIO dev.batch_sample_mode='equal-var'

# try single buffer 
RATIO=60
RUN=OneBuffer-OneHot 
python launch_context.py run_name=$RUN tasks=MT3_1cam  \
dev.one_hot=True replay.batch_size=6 replay.buffers_per_batch=1 \
framework.replay_ratio=$RATIO replay.share_across_tasks=True \
rlbench.demos=1 framework.wandb=False dev.offline=True 