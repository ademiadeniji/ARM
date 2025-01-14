# LOAD_DIR='/home/mandi/ARM/log/4tasks-sauce-target-umbrella-wine/C2FARM-SetB-Batch63-Voxel16x16-3e4/seed2/weights'
# LOAD_STEP=5000
# STEP=1000
# for TASK in  put_rubbish_in_bin # pick_up_cup phone_on_base pick_and_lift put_rubbish_in_bin
# do
#     for N_DEMO in 0 1 5 
#         do
#         taskset -c 30-48 python mt_launch.py rlbench.tasks=[${TASK}] rlbench.eval_tasks=[${TASK}] \
#         method.lr=3e-4 framework.save_freq=10 framework.log_freq=10 \
#          load=True framework.transitions_before_train=100 load_dir=${LOAD_DIR} load_step=${LOAD_STEP} framework.training_iterations=${STEP} \
#          rlbench.demos=${N_DEMO} run_name=-LOAD-4TaskSetB-Step${STEP}-${N_DEMO}demo-3e4

#         done


# done 

LOAD_DIR='/home/mandi/ARM/log/4tasks-cup-lift-phone-rubbish/C2FARM-Batch64-SetA-lr3e4-Voxel16x16/seed0/weights'
LOAD_STEP=40000
STEP=5000
CPUS=0-16
for TASK in reach_target #stack_wine take_lid_off_saucepan take_umbrella_out_of_umbrella_stand
do
    for N_DEMO in 0 #1 5 
        do
        taskset -c ${CPUS} python mt_launch.py rlbench.tasks=[${TASK}] rlbench.eval_tasks=[${TASK}] \
        method.lr=3e-4 framework.save_freq=10 framework.log_freq=10 \
         load=True framework.transitions_before_train=100 load_dir=${LOAD_DIR} load_step=${LOAD_STEP} framework.training_iterations=${STEP} \
         rlbench.demos=${N_DEMO} run_name=-LOAD-4TaskSetB-Step${STEP}-${N_DEMO}demo-3e4

        done


done 
