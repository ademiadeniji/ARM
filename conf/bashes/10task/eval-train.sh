RES=10Task-10var-Heldout-take_lid_off_saucepan/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 2500  
do
RUN=EvalOnly-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 resume_path=/shared/mandi/arm_log 
done 

# push_button
RES=10Task-10var-Heldout-push_button/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 12500 17500 
do
RUN=EvalOnly-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 resume_path=/shared/mandi/arm_log 
done

OUT=push_button
RATIO=150
RUN=Retrain-1Var-1Cam-10Buffer-Ratio${RATIO} 
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250 tasks.train_steps=8000 

RES=10Task-10var-Heldout-push_button/Retrain-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 500 1000 2500 5000 7500 
do
RUN=EvalOnly-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done

# lift 
RES=10Task-10var-Heldout-pick_and_lift/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 500 2500 7500 12500 17500 
do
RUN=EvalOnly-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 resume_path=/shared/mandi/arm_log 
done

# cup: need retrain
OUT=pick_up_cup
RES=10Task-10var-Heldout-pick_up_cup/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 12500 17500 
do
RUN=EvalOnly-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 resume_path=/shared/mandi/arm_log 
done

OUT=pick_up_cup
RATIO=150
RUN=Retrain-1Var-1Cam-10Buffer-Ratio${RATIO} 
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250 tasks.train_steps=8000 

RES=10Task-10var-Heldout-${OUT}/Retrain-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 500 1000 2500 5000 7500 
do
RUN=EvalOnly-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done

# lamp on
RES=10Task-10var-Heldout-lamp_on/1Var-1Cam-10Buffer-Ratio150-Replay_B6x10/seed0
for STEP in 500 2500 7500 12500 17500
do
RUN=EvalOnly-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 resume_path=/shared/mandi/arm_log 
done
