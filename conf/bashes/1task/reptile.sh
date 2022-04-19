RES=10Task-10var-Heldout-pick_and_lift/Reptile-NewStepLog-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000 
RATIO=30
Task=pick_and_lift
DEMO=0

# amd1
RES=10Task-10var-Heldout-pick_and_lift/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0 
STEP=20000
RATIO=30
Task=pick_and_lift

for Heldout in pick_and_lift
do 
for STEP in 12500 17500 20000 22500 25000
do
RES=10Task-10var-Heldout-${Heldout}/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
RUN=EvalReptile-Step${STEP}-NoContext
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=${RATIO} framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done 
done

RES=10Task-10var-Heldout-lamp_on/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000

Task=lamp_on
RES=10Task-10var-Heldout-${Task}/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
RATIO=60  
DEMO=2
RUN=FineTune-Reptile-Step${STEP}-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=5 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 

# for demo 1 and 3
 
STEP=20000
Task=pick_up_cup
RATIO=30

Task=push_button
RATIO=60 

Task=pick_and_lift
RATIO=60 
RES=10Task-10var-Heldout-${Task}/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000

DEMO=2
RUN=FineTune-Reptile-Step${STEP}-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=5 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 resume_path=/shared/mandi/arm_log
for SEED in 0 1 2
do

done 

Task=take_lid_off_saucepan
RES=10Task-10var-Heldout-${Task}/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed1
STEP=20000
DEMO=2
for SEED in 0 1 2
do
RUN=FineTune-Reptile-Step${STEP}-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=5 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 
done  

# Eval

RES=10Task-10var-Heldout-lamp_on/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0

# amd1
RES=10Task-10var-Heldout-pick_and_lift/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 5000 7500 10000 12500 15000 17500 20000 22500 25000 
do
RUN=EvalReptile-Step${STEP}-NoContext
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=${RATIO} framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done 

RES=10Task-10var-Heldout-pick_up_cup/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 1000 5000 10000 15000
do
RUN=EvalReptile-Step${STEP}-NoContext
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=${RATIO} framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done 


resume_path=/shared/mandi/arm_log 