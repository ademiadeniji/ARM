OUT=push_button
RATIO=150
K=5
EPS=0.1
RUN=Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_reptile=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0] \
tasks.train_steps=50000 

# single task 
Task=pick_up_cup
RATIO=150
K=5
EPS=0.1
RUN=Reptile-Kstep${K}-Eps${EPS}-15Var-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=15 run_name=${RUN} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=100 \
replay.batch_size=15 framework.log_freq=100  \
dev.use_reptile=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0] \
tasks.train_steps=20000 tasks=Single_1cam tasks.all_tasks=[${Task}] 