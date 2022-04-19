OUT=stack_wine
RATIO=300 
RUN=1Var-1Cam-1Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py rlbench.num_vars=1 run_name=$RUN tasks.heldout=${OUT} \
framework.replay_ratio=$RATIO mt_only=True replay.share_across_tasks=True \
replay.batch_size=300 replay.buffers_per_batch=1 rlbench.demos=5

# 10-buff
OUT=stack_wine
OUT=put_rubbish_in_bin
OUT=take_lid_off_saucepan
OUT=push_button
OUT=pick_up_cup
RATIO=300 
RATIO=100
OUT=stack_wine
RUN=1Var-1Cam-10Buffer-Ratio${RATIO} 
taskset -c $CPUS python launch_context.py rlbench.num_vars=1 run_name=$RUN tasks.heldout=${OUT} \
framework.replay_ratio=$RATIO mt_only=True \
replay.batch_size=30 replay.buffers_per_batch=10 rlbench.demos=5 \
 framework.save_freq=100 framework.log_freq=100 


# batch 150 

OUT=lamp_on

OUT=pick_and_lift
RATIO=150
RUN=Demo1-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250 rlbench.demos=1


# reptile 
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

# try 1 task per batch
OUT=pick_and_lift
RATIO=150
RUN=1TaskBatch-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250 replay.num_tasks_per_batch=1



# Eval on 10 trained tasks  
OUT=take_lid_off_saucepan
RES=10Task-10var-Heldout-take_lid_off_saucepan/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=1000  
RATIO=1 
for STEP in  5000 10000 20000 25000 
do
RUN=EvalOnly-${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=${OUT}  \
framework.replay_ratio=${RATIO} rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=10 resume_path=/shared/mandi/arm_log 
done 
