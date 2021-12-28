# dVAE, 10 task per batch, pabamd2
OUT=take_lid_off_saucepan
DIM=16
HID=2048
RUN=1Cam-dVAE-3frameStack-10TaskBatch-Ratio12
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks.heldout=${OUT} \
replay.batch_size=6 replay.buffers_per_batch=15 replay.num_tasks_per_batch=10 framework.replay_ratio=12 \
replay.replay_size=1000000 framework.training_iterations=100000 framework.log_freq=500 \
tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID}  