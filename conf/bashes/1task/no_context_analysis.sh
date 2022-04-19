######### 0 demo group:

Task=pick_up_cup
RES=10Task-10var-Heldout-pick_up_cup/1Var-1Cam-10Buffer-Ratio300-Replay_B30x10/seed0
STEP=25000
RATIO=30 
 
Task=take_lid_off_saucepan
RES=10Task-10var-Heldout-take_lid_off_saucepan/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=25000
RATIO=60
for STEP in 1000 5000 10000 
do 
DEMO=0
RUN=Analysis-${STEP}-FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=10000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 resume_path=/shared/mandi/arm_log 
done

Task=pick_and_lift
RES=10Task-10var-Heldout-pick_and_lift/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000 
RATIO=30

Task=push_button
RES=10Task-10var-Heldout-push_button/1Var-1Cam-10Buffer-Ratio300-Replay_B30x10/seed0
STEP=15000 
RATIO=60 

Task=lamp_on
RES=10Task-10var-Heldout-lamp_on/1Var-1Cam-10Buffer-Ratio150-Replay_B6x10/seed0
STEP=20000
RATIO=60
 
for STEP in 1000 5000 10000 
do 
for SEED in 0 1 2 
do 
DEMO=0
RUN=Analysis-${STEP}-FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=10000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 resume_path=/shared/mandi/arm_log 
done
done 

######### >1 demo 

Task=stack_wine
RES=10Task-10var-Heldout-stack_wine/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=30000

Task=put_rubbish_in_bin
RES=10Task-10var-Heldout-put_rubbish_in_bin/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=25000 
RATIO=30

# finetune no-demo run
Task=pick_and_lift
# amd1: 0 and 20 demo
RES=10Task-10var-Heldout-pick_and_lift/Demo0-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0 
RES=10Task-10var-Heldout-pick_and_lift/Demo20-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0 
# amd2: 1demo
Task=pick_and_lift
RES=10Task-10var-Heldout-pick_and_lift/Demo1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000 
DEMO=0
RATIO=30 
for SEED in 0 1 2
do
RUN=Analysis-FineTune-1DemoTrain-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=10000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1  
done 
### get eval-only data


Task=lamp_on
RES=10Task-10var-Heldout-lamp_on/1Var-1Cam-10Buffer-Ratio150-Replay_B6x10/seed0

Task=pick_and_lift
RES=10Task-10var-Heldout-pick_and_lift/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0


for STEP in 1000 5000 10000 15000 20000
Task=pick_and_lift
RES=10Task-10var-Heldout-pick_and_lift/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0

for STEP in 1000 
do
RUN=EvalOnly-Step${STEP}-NoContext
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=${RATIO} framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 resume_path=/shared/mandi/arm_log tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30
done


# special case for pick_up_cup: use ckpt saved on pabti5! 
Task=pick_up_cup
RES=10Task-10var-Heldout-pick_up_cup/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in 25000 
do
RUN=EvalOnly-Step${STEP}-NoContext
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=${RATIO} framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 resume_path=/shared/mandi/arm_log 
done

# for push button: need to stitch some checkpoints 
Task=push_button
RES=10Task-10var-Heldout-push_button/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
for STEP in  20000  
do
RUN=EvalOnly-Step${STEP}-NoContext
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=${RATIO} framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 resume_path=/shared/mandi/arm_log 
done

# Eval 11 task but no-demo run: [amd2]
Task=pick_and_lift
RES=11Task-11var-NoDemo-pick_and_lift/1Var-NoContext-Train0Demo-Ratio150-Replay_B13x11/seed1

Task=lamp_on
RES=11Task-11var-NoDemo-lamp_on/1Var-NoContext-Train0Demo-Ratio150-Replay_B13x11/seed0

Task=take_lid_off_saucepan
RES=11Task-11var-NoDemo-take_lid_off_saucepan/1Var-NoContext-Train0Demo-Ratio150-Replay_B13x11/seed1
for STEP in 1000 5000 10000 15000 20000  
do
RUN=EvalNoDemo-${Task}-Step${STEP}-NoContext
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

# eval the Reptile checkpoints: should times train steps by 5 if not new-log 
RES=10Task-10var-Heldout-lamp_on/Reptile-NewStepLog-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
Task=lamp_on

RES=10Task-10var-Heldout-take_lid_off_saucepan/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0 
Task=take_lid_off_saucepan
for STEP in 0 1000 5000 10000 15000 20000
do
RUN=EvalReptile-${Task}-Step${STEP}-NoContext
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
