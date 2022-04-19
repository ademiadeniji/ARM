# use MT11 config 
NoDemo=[lamp_on]
RATIO=150
RUN=1Var-NoContext-Train0Demo-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} tasks=MT11_1cam \
tasks.no_demo_tasks=${NoDemo} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=11 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=13 framework.log_freq=250 