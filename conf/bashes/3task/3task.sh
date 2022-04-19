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
RUN=OneBuffer-OneHot-${RATIO}
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  \
dev.one_hot=True replay.batch_size=60 replay.buffers_per_batch=1 \
framework.replay_ratio=$RATIO replay.share_across_tasks=True \
rlbench.demos=5 

# resume from good pick-up-cup 
DIM=16
HID=2048 
RATIO=60
RES=pick_up_cup-20var/Scratch-1Cam-dVAE-3frameStack-Ratio12-Demo1-Replay_B6x10/seed1
STEP=9999
DEMO=5 
RUN=Resume-PickUpCheckpoint-Step${STEP}-EqualTaskBatch
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 \
run_name=$RUN tasks=MT3_1cam \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
 rlbench.demos=${DEMO} framework.ckpt_eval=True  \
 resume=True resume_run=${RES} resume_step=${STEP} framework.replay_ratio=${RATIO} \
replay.batch_size=6 replay.buffers_per_batch=10 dev.batch_sample_mode='equal-task' framework.save_freq=100


RUN=OneHot-SumLoss-NoWeights
RATIO=10
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  dev.one_hot=True replay.batch_size=10 \
replay.buffers_per_batch=6 framework.replay_ratio=$RATIO rlbench.demos=5 replay.num_tasks_per_batch=3  \
tasks.train_steps=20000 dev.batch_sample_mode='equal-task' replay.buffers_per_batch=6

# equal-task 
RATIO=60
RUN=OneHot-EqualTask
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  dev.one_hot=True  \
replay.batch_size=3 replay.buffers_per_batch=30 framework.replay_ratio=$RATIO rlbench.demos=5 replay.num_tasks_per_batch=3  \
tasks.train_steps=20000 dev.batch_sample_mode='equal-task'  

RATIO=60
GSTEP=3
RUN=OneHot-EqualTask-GradAccum${GSTEP}-1TaskPerBatch
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  dev.one_hot=True  \
replay.batch_size=6 replay.buffers_per_batch=15 \
framework.replay_ratio=$RATIO rlbench.demos=5 \
replay.num_tasks_per_batch=3 tasks.train_steps=50000 \
dev.batch_sample_mode='equal-task' dev.grad_accum=${GSTEP} framework.num_tasks_per_batch=1

# use dVAE
RATIO=60
GSTEP=5
DIM=16
HID=2048 
RUN=OneHot-EqualTask-GradAccum${GSTEP}-1TaskPerBatch
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
replay.batch_size=6 replay.buffers_per_batch=15 \
framework.replay_ratio=$RATIO rlbench.demos=5 \
tasks.train_steps=100000 \
dev.batch_sample_mode='equal-task' dev.grad_accum=${GSTEP} \
replay.num_tasks_per_batch=1
