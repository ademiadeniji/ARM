# hw2: MT 4 task

# rtxl1
CPUS=0-16
RATIO=64
RUN=Hw2-OneBuffer
taskset -c $CPUS python launch_context.py run_name=${RUN} mt_only=True \
tasks=['push_button','pick_and_lift','reach_target','take_lid_off_saucepan'] rlbench.num_vars=1 \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
replay.share_across_tasks=True replay.batch_size=128 rlbench.demos=10