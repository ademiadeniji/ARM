# 1hot for now, fewer train steps
RATIO=10
K=5
EPS=0.5
RUN=OneHot-Reptile-Kstep${K}-Eps${EPS}-LongAnneal
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  \
dev.one_hot=True replay.batch_size=6 replay.buffers_per_batch=10 \
framework.replay_ratio=$RATIO rlbench.demos=5 \
dev.use_reptile=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0] \
tasks.train_steps=100000
 
 # sum loss
RATIO=10
RUN=OneHot-SumLoss
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  \
dev.one_hot=True replay.batch_size=10 replay.buffers_per_batch=6 \
framework.replay_ratio=$RATIO rlbench.demos=5 replay.num_tasks_per_batch=3 

# normalize buffer std 
RATIO=10
RUN=OneHot-Buffer-Norm-ByBuffer
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=MT3_1cam  \
dev.one_hot=True replay.batch_size=10 replay.buffers_per_batch=6 \
framework.replay_ratio=$RATIO rlbench.demos=5 \
replay.num_tasks_per_batch=3 dev.normalize_reward='by-buffer'
