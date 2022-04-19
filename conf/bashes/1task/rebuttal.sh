# scratch
CPUS=0-64
DEMO=0
TASK=[reach_target]
TASK=[open_door]

DEMO=0
TASK=[take_umbrella_out_of_umbrella_stand]
for SEED in 1 2 
do
taskset -c $CPUS python launch_context.py run_name=Scratch-NoContext-1Buffer-Ratio60-Demo${DEMO} \
mt_only=True tasks=Single_1cam framework.replay_ratio=60 rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 framework.num_log_episodes=10 \
framework.eval_envs=4 rlbench.num_vars=1 replay.replay_size=100000 tasks.all_tasks=$TASK \
framework.training_iterations=5000  tasks.num_vars=1 
done

# Finetune existing MT10 agents on new tasks
TASK=[take_umbrella_out_of_umbrella_stand]


DEMO=1
TASK=[close_drawer]
TASK=[take_umbrella_out_of_umbrella_stand]
DEMO=1
for SEED in 1 2  
do
taskset -c $CPUS python launch_context.py run_name=FineTune-NoContext-1Buffer-Ratio60-Demo$DEMO mt_only=True \
tasks=Single_1cam tasks.all_tasks=$TASK framework.replay_ratio=60 rlbench.demos=$DEMO framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=100000 \
resume=True resume_run=10Task-10var-Heldout-stack_wine/1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0 resume_step=35000 \
rlbench.num_vars=-1 resume_path=/shared/mandi/arm_log framework.training_iterations=5000  tasks.num_vars=1 
done 
# reptile FT

TASK=[take_umbrella_out_of_umbrella_stand]
TASK=[close_drawer]
DEMO=0
for SEED in 1 2 3
do
taskset -c $CPUS python launch_context.py run_name=FineTune-Reptile-Ratio150-Demo$DEMO \
 mt_only=True tasks=Single_1cam tasks.all_tasks=$TASK framework.replay_ratio=60 \
 rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=60 replay.buffers_per_batch=1 \
 framework.num_log_episodes=10 framework.eval_envs=5 replay.replay_size=100000 \
 resume=True resume_run=10Task-10var-Heldout-take_lid_off_saucepan/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed1 \
 resume_step=20000 framework.training_iterations=5000 tasks.num_vars=1 resume_path=/shared/mandi/arm_log
 done

 # PEARL FT 
TASK=[take_umbrella_out_of_umbrella_stand]
for DEMO in 0 1 
do
for SEED in 1 2 3
do
taskset -c $CPUS python launch_context.py run_name=FineTune-Pearl-Step10000-Ratio30-Demo$DEMO tasks=Single_1cam \
tasks.all_tasks=$TASK framework.replay_ratio=30 rlbench.demos=$DEMO replay.batch_size=10 \
replay.buffers_per_batch=1 framework.eval_envs=5 framework.train_envs=0 replay.replay_size=100000 \
resume=True resume_run=10Task-10var-Heldout-take_lid_off_saucepan/Pearl-3Context-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0 \
resume_step=10000 dev.use_pearl=True contexts.loss_mode=pearl dev.pearl_context_size=3 method.lr=0 method.emb_lr=0 \
framework.transitions_before_train=15000 framework.num_log_episodes=30 \
resume_path=/shared/mandi/arm_log dev.pearl_onpolicy_context=False tasks.train_steps=2 rlbench.num_vars=1 \
framework.training_iterations=5000
done 
done 

# single-task, 10vars 
# pearl: pabamd1

taskset -c $CPUS python launch_context.py rlbench.num_vars=10 run_name=Pearl-3Context-DemoCon-10Var-10Buffer-Ratio150 \
tasks=Single_1cam tasks.all_tasks=[push_button] framework.replay_ratio=150 replay.buffers_per_batch=10 \
rlbench.demos=5 framework.save_freq=250 replay.batch_size=6 framework.log_freq=250 \
dev.use_pearl=True contexts.loss_mode=pearl tasks.train_steps=50000 dev.pearl_context_size=3 dev.pearl_onpolicy_context=False

# mt
taskset -c $CPUS python launch_context.py mt_only=True rlbench.num_vars=10 run_name=10Var-10Buffer-Ratio150 \
tasks=Single_1cam tasks.all_tasks=[push_button] framework.replay_ratio=150 replay.buffers_per_batch=10 \
rlbench.demos=5 framework.save_freq=250 replay.batch_size=6 framework.log_freq=250 

# reptile
taskset -c $CPUS python launch_context.py mt_only=True rlbench.num_vars=10 run_name=Reptile-Kstep5-Eps0.1-10Var-10Buffer-Ratio150 \
tasks=Single_1cam tasks.all_tasks=[push_button] replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=6 framework.log_freq=250 dev.use_reptile=True dev.reptile_k=5 dev.reptile_eps=[0.1,0] tasks.train_steps=50000



# Finetune 10var

DEMO=0
TASK=[push_button]

for SEED in 1 2 3
do
taskset -c $CPUS python launch_context.py run_name=FineTune-NewVar-Demo$DEMO mt_only=True \
tasks=Single_1cam tasks.all_tasks=$TASK framework.replay_ratio=60 rlbench.demos=$DEMO framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=100000 \
resume=True resume_run=push_button-18var/10Var-10Buffer-Ratio150-Replay_B6x10/seed0 resume_step=9999 \
rlbench.num_vars=5 framework.training_iterations=5000 dev.handpick=[10,11,12,13,14]
done 

# try different lr and transitions before train
DEMO=0
TASK=[push_button]
for LR in 5e-4 7e-4 5e-5 #1e-4 3e-4 #
do
for BEFORE in 60 100 120
do
taskset -c $CPUS python launch_context.py run_name=FineTune-NewVar-LR${LR}-Before$BEFORE mt_only=True \
tasks=Single_1cam tasks.all_tasks=$TASK framework.replay_ratio=60 rlbench.demos=$DEMO framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=100000 \
resume=True resume_run=push_button-18var/10Var-10Buffer-Ratio150-Replay_B6x10/seed0 resume_step=9999 \
rlbench.num_vars=5 framework.training_iterations=1000 dev.handpick=[10,11,12,13,14] method.lr=$LR \
framework.transitions_before_train=$BEFORE
done 
done

# reptile

for SEED in 1 2 3
do
taskset -c $CPUS python launch_context.py run_name=FineTune-Reptile-NewVar-Demo$DEMO mt_only=True \
tasks=Single_1cam tasks.all_tasks=$TASK framework.replay_ratio=60 rlbench.demos=$DEMO framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 framework.num_log_episodes=10 framework.eval_envs=4 \
replay.replay_size=100000 \
resume=True resume_run=push_button-18var/Reptile-Kstep5-Eps0.1-10Var-10Buffer-Ratio150-Replay_B6x10/seed1 resume_step=750 \
rlbench.num_vars=5 framework.training_iterations=5000 dev.handpick=[10,11,12,13,14]
done 

# pearl on 10var

TASK=[push_button]
DEMO=0
for SEED in 1 2 3
do
taskset -c $CPUS python launch_context.py run_name=FineTune-Pearl-Step10000-Ratio30-Demo$DEMO tasks=Single_1cam \
tasks.all_tasks=$TASK framework.replay_ratio=30 rlbench.demos=$DEMO replay.batch_size=10 \
replay.buffers_per_batch=1 framework.eval_envs=5 framework.train_envs=0 replay.replay_size=100000 \
resume=True resume_run=push_button-18var/Pearl-3Context-DemoCon-10Var-10Buffer-Ratio150-Replay_B6x10/seed0 resume_step=10000 \
resume_step=10000 dev.use_pearl=True contexts.loss_mode=pearl dev.pearl_context_size=3 method.lr=0 method.emb_lr=0 \
framework.transitions_before_train=8000 framework.num_log_episodes=30 \
dev.pearl_onpolicy_context=False tasks.train_steps=2 rlbench.num_vars=5 dev.handpick=[10,11,12,13,14] \
framework.training_iterations=5000
done 

DEMO=0
TASK=[push_button]
for SEED in 1 2  3
do
taskset -c $CPUS python launch_context.py run_name=Scratch-NoContext-1Buffer-Ratio60-Demo${DEMO} \
mt_only=True tasks=Single_1cam framework.replay_ratio=60 rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 framework.num_log_episodes=10 \
framework.eval_envs=4 replay.replay_size=100000 tasks.all_tasks=$TASK \
framework.training_iterations=5000  rlbench.num_vars=5 dev.handpick=[10,11,12,13,14] 
done

 
