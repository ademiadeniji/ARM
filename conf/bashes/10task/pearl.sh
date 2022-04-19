OUT=lamp_on
OUT=take_lid_off_saucepan
OUT=push_button

RATIO=150
NUM_C=3
RUN=Pearl-${NUM_C}Context-DemoCon-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_pearl=True contexts.loss_mode='pearl' \
tasks.train_steps=50000  dev.pearl_context_size=${NUM_C} dev.pearl_onpolicy_context=False

 

OUT=push_button
OUT=lamp_on
OUT=pick_up_cup
OUT=take_lid_off_saucepan
OUT=put_rubbish_in_bin
RATIO=150
NUM_C=3
RUN=Pearl-${NUM_C}Context-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_pearl=True contexts.loss_mode='pearl' \
tasks.train_steps=50000  dev.pearl_context_size=${NUM_C}

# multi-var 
Task=pick_up_cup
RATIO=10
NUM_C=3
RUN=Pearl-${NUM_C}Context-Ratio${RATIO}
taskset -c $CPUS python launch_context.py \
rlbench.num_vars=-1 run_name=${RUN}  framework.replay_ratio=$RATIO \
replay.buffers_per_batch=15 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=6 framework.log_freq=100  \
dev.use_pearl=True contexts.loss_mode='pearl' \
dev.pearl_context_size=${NUM_C} tasks=Single_1cam tasks.all_tasks=[${Task}] 


# fine-tune: NEW: rollout a bunch of eval steps , no ckpt eval!!
Task=pick_up_cup
NUM_C=3
RES=10Task-10var-Heldout-${Task}/Pearl-3Context-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000
DEMO=0

for DEMO in 0 1 3
do
RUN=FineTune-Pearl-Step${STEP}-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} \
replay.batch_size=10 replay.buffers_per_batch=1 \
framework.eval_envs=5 framework.train_envs=0 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
dev.use_pearl=True contexts.loss_mode='pearl' dev.pearl_context_size=${NUM_C} \
method.lr=0 method.emb_lr=0 framework.transitions_before_train=15000 \
framework.num_log_episodes=30 dev.pearl_onpolicy_context=False \
tasks.train_steps=2 rlbench.num_vars=1  
 
done

resume_path=/shared/mandi/arm_log 

Task=push_button 
RES=10Task-10var-Heldout-push_button/Pearl-3Context-DemoCon-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0


Task=take_lid_off_saucepan
NUM_C=3
RES=10Task-10var-Heldout-take_lid_off_saucepan/Pearl-3Context-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=10000
DEMO=0

Task=lamp_on
NUM_C=6
RES=10Task-10var-Heldout-lamp_on/Pearl-6Context-DemoCon-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=20000 
DEMO=0
 
Task=pick_and_lift
RES=10Task-10var-Heldout-pick_and_lift/Pearl-3Context-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed1 

# Eval:
# need to set framework.train_envs==0
 

# rtxl2 #offline!!, praying rtxl1 doesn't crash
RES=10Task-10var-Heldout-pick_and_lift/Pearl-3Context-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed3
for STEP in 500 1000 2500 5000 7500 10000 12500 15000 17500 20000
do
for Task in push_button take_lid_off_saucepan lamp_on pick_up_cup pick_and_lift phone_on_base stack_wine open_door take_usb_out_of_computer put_rubbish_in_bin reach_target
do
RUN=EvalPearl-Step${STEP}-${Task} 
taskset -c $CPUS python launch_context.py run_name=$RUN \
tasks.heldout='' framework.replay_ratio=10 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.num_log_episodes=30 framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
dev.use_pearl=True contexts.loss_mode='pearl' dev.pearl_context_size=3 \
rlbench.num_vars=1  tasks.train_steps=1  resume=True resume_run=${RES} resume_step=${STEP} \
framework.transitions_before_train=10 rlbench.demos=0 framework.train_envs=0 \
tasks=Single_1cam tasks.all_tasks=[${Task}] 
done
done

# rtxls1
RES=10Task-10var-Heldout-lamp_on/Pearl-6Context-DemoCon-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0 
for STEP in 500 1000 2500 5000 7500 10000 12500 15000 17500 20000 
do
for Task in push_button take_Slid_off_saucepan lamp_on pick_up_cup pick_and_lift phone_on_base stack_wine open_door take_usb_out_of_computer put_rubbish_in_bin reach_target
do
RUN=EvalPearl-Step${STEP}-${Task} 
taskset -c $CPUS python launch_context.py run_name=$RUN \
tasks.heldout='' framework.replay_ratio=10 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.num_log_episodes=30 framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
dev.use_pearl=True contexts.loss_mode='pearl' dev.pearl_context_size=${NUM_C} \
rlbench.num_vars=1  tasks.train_steps=1  resume=True resume_run=${RES} resume_step=${STEP} \
framework.transitions_before_train=10 rlbench.demos=0 framework.train_envs=0 \
tasks=Single_1cam tasks.all_tasks=[${Task}] 
done
done



# eval! 