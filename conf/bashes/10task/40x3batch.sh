# one hot only for now 

OUT=take_lid_off_saucepan
RUN=1Cam-OntHot-10TaskBatch-Ratio12 
taskset -c $CPUS python launch_context.py rlbench.num_vars=-1 run_name=$RUN tasks.heldout=${OUT} \
dev.one_hot=True replay.batch_size=3 replay.buffers_per_batch=40 replay.num_tasks_per_batch=10 \
framework.replay_ratio=12 