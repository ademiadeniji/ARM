OUT=pick_up_cup
RATIO=150
NUM_C=5 
RUN=Pearl-${NUM_C}Context-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_pearl=True contexts.loss_mode='pearl' \
tasks.train_steps=50000  dev.pearl_context_size=${NUM_C} 