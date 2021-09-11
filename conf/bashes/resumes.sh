# RUN=7tasks-cup-lift-phone-rubbish-sauce-target-umbrella/C2FARM-Fresh-Batch126-lr5e-4/seed0
# STEP=24000
# for DEMO in 10 0 
# do

# for BEFORE in 200 #100
# do
#     #python launch_multitask.py tasks=['stack_wine']  resume=True run_name=Resume-7Task-Batch64-lr1e4-${DEMO}Demo-${BEFORE}before \
#     #framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     #resume_run=${RUN} resume_step=${STEP} \
#     rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=1e-4; 

#     python launch_multitask.py tasks=['stack_wine']  resume=True run_name=Resume-7Task-Batch64-lr2e4-${DEMO}Demo-${BEFORE}before \
#     framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     resume_run=${RUN} resume_step=${STEP} \
#     rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=2e-4; 

#     python launch_multitask.py tasks=['stack_wine']  resume=True run_name=Resume-7Task-Batch64-lr3e4-${DEMO}Demo-${BEFORE}before \
#     framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     resume_run=${RUN} resume_step=${STEP} \
#     rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=3e-4; 


#     done
# done

# RUN=7tasks-cup-lift-rubbish-sauce-target-umbrella-wine/C2FARM-Fresh-Batch63-lr5e4-Network3x16/seed0
# STEP=26000
# TASK=phone_on_base
# for DEMO in 10 1 0 
# do

# for BEFORE in 200 #100
# do
#     # python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr1e4-${DEMO}Demo-${BEFORE}before \
#     # framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     # resume_run=${RUN} resume_step=${STEP} \
#     # rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=1e-4; 

#     # python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr2e4-${DEMO}Demo-${BEFORE}before \
#     # framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     # resume_run=${RUN} resume_step=${STEP} \
#     # rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=2e-4; 

#     python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-3x16Network-7Task-Batch64-lr3e4-${DEMO}Demo-${BEFORE}before \
#     framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     resume_run=${RUN} resume_step=${STEP} \
#     rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=3e-4 \
#     method.voxel_sizes=[16,16,16] method.bounds_offset=[0.15,0.15] ;

#     python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-3x16Network-7Task-Batch64-lr5e4-${DEMO}Demo-${BEFORE}before \
#     framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     resume_run=${RUN} resume_step=${STEP} \
#     rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=5e-4 \
#     method.voxel_sizes=[16,16,16] method.bounds_offset=[0.15,0.15] ;

#     done
# done

# change resume_path=/shared/mandi/rlbench_log to load from different path 
# RUN=C2FARM-OneBuffer-Batch63-lr5e4/seed0
# STEP=19900
# TASK=stack_wine 
# for DEMO in 1 0 10
# do  
#  for BEFORE in 200 100
# do
#     # python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr1e4-${DEMO}Demo-${BEFORE}before \
#     # framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     # resume_run=${RUN} resume_step=${STEP} \
#     # rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=1e-4; 

#     # python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr2e4-${DEMO}Demo-${BEFORE}before \
#     # framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     # resume_run=${RUN} resume_step=${STEP} \
#     # rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=2e-4; 
#     python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr5e4-${DEMO}Demo-${BEFORE}before \
#     framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     resume_run=${RUN} resume_step=${STEP} \
#     rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=5e-4  resume_path=/shared/mandi/rlbench_log;

#     python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr3e4-${DEMO}Demo-${BEFORE}before \
#     framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
#     resume_run=${RUN} resume_step=${STEP} \
#     rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=3e-4 resume_path=/shared/mandi/rlbench_log;


#     done
# done


RUN=7tasks-lift-phone-rubbish-sauce-target-umbrella-wine/C2FARM-OneBuffer-Batch63-lr5e4/seed0
STEP=49900
TASK=pick_up_cup 
for DEMO in 10 1 0
do  
 for BEFORE in 200 100
do
    # python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr1e4-${DEMO}Demo-${BEFORE}before \
    # framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    # resume_run=${RUN} resume_step=${STEP} \
    # rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=1e-4; 

    # python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr2e4-${DEMO}Demo-${BEFORE}before \
    # framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    # resume_run=${RUN} resume_step=${STEP} \
    # rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=2e-4; 
    python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr5e4-${DEMO}Demo-${BEFORE}before \
    framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    resume_run=${RUN} resume_step=${STEP} \
    rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=5e-4 

    python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-7Task-Batch64-lr3e4-${DEMO}Demo-${BEFORE}before \
    framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    resume_run=${RUN} resume_step=${STEP} \
    rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=3e-4  


    done
done


TASK=take_lid_off_saucepan
for DEMO in 0 1  
do  
 for BEFORE in 200 100
do
    python launch_multitask.py tasks=[$TASK]  resume=True run_name=Resume-14Task-Batch64-lr5e4-${DEMO}Demo-${BEFORE}before \
    framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    framework.log_freq=50 \
    resume_run=${RUN} resume_step=${STEP} rlbench.demo_path=/shared/mandi/all_rlbench_data \
    rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=5e-4 

    done
done

for TASK in take_lid_off_saucepan reach_target press_switch; do for DEMO in 0 1; do    for BEFORE in 50 100 200; do     python launch_context.py tasks=[$TASK]  run_name=Baseline-NoContext-lr5e4 mt_only=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE}     rlbench.demo_path=/home/mandi/front_rlbench_data   rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=5e-4 framework.log_freq=50  ;      done; done; done