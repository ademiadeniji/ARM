Task=push_button
RATIO=30
DEMO=0
RUN=2MLP-CLIP-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=$RATIO rlbench.demos=${DEMO}  replay.batch_size=60 \
replay.buffers_per_batch=1 rlbench.num_vars=1 dev.one_hot=True \
tasks.train_steps=100000 replay.replay_size=100000 dev.clip_reward="robot gripper tip touch square button"
