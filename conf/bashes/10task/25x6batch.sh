# one-hot, 10 task per batch, pabrtxl1
OUT=push_button
RUN=1Cam-OntHot-10TaskBatch
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks.heldout=${OUT} \
dev.one_hot=True replay.batch_size=6 replay.buffers_per_batch=25 replay.num_tasks_per_batch=10 framework.replay_ratio=12 \
replay.replay_size=1000000 framework.training_iterations=100000 framework.log_freq=500




