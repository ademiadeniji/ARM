

Task=pick_up_cup

Task=put_rubbish_in_bin 

Task=push_button  
Task=stack_wine 

RATIO=60
Task=lamp_on
DEMO=2
RUN=Scratch-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
for S in 0 1 2
do
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
    tasks=Single_1cam tasks.all_tasks=[${Task}] \
    framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
    replay.batch_size=60 replay.buffers_per_batch=1 \
    framework.num_log_episodes=10 framework.eval_envs=4 rlbench.num_vars=1 \
    replay.replay_size=100000 
done

Task=put_rubbish_in_bin 
RATIO=60

# also try 30 
Task=stack_wine
RATIO=30
Task=pick_up_cup
Task=pick_and_lift
RATIO=60 
for DEMO in 2 2 # 1 2 3 4 # 5 6 7 8 9 10
do 
    RUN=Scratch-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
    taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
    tasks=Single_1cam tasks.all_tasks=[${Task}] \
    framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
    replay.batch_size=60 replay.buffers_per_batch=1 \
    framework.num_log_episodes=10 framework.eval_envs=4 \
    replay.replay_size=100000 rlbench.num_vars=1
done

# fine-tune from 10task
Task=pick_up_cup
RES=10Task-10var-Heldout-pick_up_cup/1Var-1Cam-10Buffer-Ratio300-Replay_B30x10/seed0
STEP=25000

# load from shared
Task=stack_wine
RES=10Task-10var-Heldout-stack_wine/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=30000


Task=put_rubbish_in_bin
RES=10Task-10var-Heldout-put_rubbish_in_bin/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=25000 
RATIO=30

Task=take_lid_off_saucepan
RES=10Task-10var-Heldout-take_lid_off_saucepan/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=25000
for STEP in 1000 5000 10000 15000 20000 
do 
Task=take_lid_off_saucepan
RES=10Task-10var-Heldout-take_lid_off_saucepan/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
DEMO=0
RATIO=60
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


# freeze layers 0 or 1
FREEZE=1
for SEED in 0 1 2 
do 
RUN=FineTune-NoContext-Freeze${FREEZE}-1Buffer-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
    tasks=Single_1cam tasks.all_tasks=[${Task}] \
    framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
    replay.batch_size=60 replay.buffers_per_batch=1 \
    framework.num_log_episodes=10 framework.eval_envs=4 \
    replay.replay_size=10000 resume=True resume_run=${RES} resume_step=${STEP} \
    rlbench.num_vars=1 resume_path=/shared/mandi/arm_log resume_freeze=[${FREEZE}]
done 

Task=stack_wine
RES=10Task-10var-Heldout-stack_wine/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=30000
RATIO=60 
for DEMO in 2 2 #0 1 2 3 4 5 # 6 7 8 9 10
do 
    RUN=FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
    taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
    tasks=Single_1cam tasks.all_tasks=[${Task}] \
    framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
    replay.batch_size=60 replay.buffers_per_batch=1 \
    framework.num_log_episodes=10 framework.eval_envs=4 \
    replay.replay_size=10000 resume=True resume_run=${RES} resume_step=${STEP} \
    rlbench.num_vars=1 resume_path=/shared/mandi/arm_log 
done
 
Task=pick_up_cup
RES=10Task-10var-Heldout-pick_up_cup/1Var-1Cam-10Buffer-Ratio300-Replay_B30x10/seed0
STEP=25000
RATIO=30
for DEMO in 1 #0 1 2 3 4 5 # 6 7 8 9 10
do 
    RUN=FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
    taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
    tasks=Single_1cam tasks.all_tasks=[${Task}] \
    framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
    replay.batch_size=60 replay.buffers_per_batch=1 \
    framework.num_log_episodes=10 framework.eval_envs=4 \
    replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
    rlbench.num_vars=1 resume_path=/shared/mandi/arm_log
done


Task=push_button
RES=10Task-10var-Heldout-push_button/1Var-1Cam-10Buffer-Ratio300-Replay_B30x10/seed0
# NEW: try a dfferent ckpt! 

Task=push_button
RES=10Task-10var-Heldout-push_button/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000 
RATIO=60
DEMO=0 
RUN=FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=-1 resume_path=/shared/mandi/arm_log

STEP=15000 
RATIO=60
for DEMO in 0  # 6 7 8 9 10
do 
    RUN=FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
    taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
    tasks=Single_1cam tasks.all_tasks=[${Task}] \
    framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
    replay.batch_size=60 replay.buffers_per_batch=1 \
    framework.num_log_episodes=10 framework.eval_envs=4 \
    replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
    rlbench.num_vars=-1 resume_path=/shared/mandi/arm_log
done

 
Task=pick_and_lift
RES=10Task-10var-Heldout-pick_and_lift/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000 

DEMO=1

DEMO=0 
RATIO=30 
for SEED in 0 1 2
do
RUN=FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 resume_path=/shared/mandi/arm_log 
done

Task=lamp_on
RES=10Task-10var-Heldout-lamp_on/1Var-1Cam-10Buffer-Ratio150-Replay_B6x10/seed0

RATIO=60 
DEMO=2
STEP=20000
RUN=FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=5 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 resume_path=/shared/mandi/arm_log 

RATIO=60
DEMO=0
for DEMO in 1 1
do
RUN=FineTune-NoContext-1Buffer-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 resume_path=/shared/mandi/arm_log 
done
 