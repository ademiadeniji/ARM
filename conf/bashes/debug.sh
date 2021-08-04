#LOAD_DIR=/home/mandi/ARM/log/7tasks-cup-lift-rubbish-sauce-target-umbrella-wine/C2FARM-Hard-Batch126-Voxel16x16-5e4/seed0/weights
LOAD_DIR=/home/mandi/ARM/log/8tasks-cup-lift-phone-rubbish-sauce-target-umbrella-wine/C2FARM-Batch128-lr3e4-Fill100-Demo5/seed0/weights
LOAD_STEP=29900
STEP=1000
CPUS=0-30
RUN=EvalLOAD-8Task
for TASK in phone_on_base #stack_wine take_lid_off_saucepan take_umbrella_out_of_umbrella_stand
do
    for N_DEMO in 5 1 10
        do
        taskset -c ${CPUS} python mt_launch.py rlbench.tasks=[${TASK}] rlbench.eval_tasks=[${TASK}] \
                            method.lr=3e-4 framework.save_freq=100 framework.log_freq=10 \
                                     load=True framework.transitions_before_train=100 load_dir=${LOAD_DIR} load_step=${LOAD_STEP} framework.training_iterations=${STEP} \
                                              rlbench.demos=${N_DEMO} run_name=-${RUN}-Step${STEP}-Fill100-${N_DEMO}demo-3e4 #rlbench.eval_tasks='${8_tasks}' framework.n_eval=4

        #taskset -c ${CPUS} python mt_launch.py rlbench.tasks=[${TASK}] rlbench.eval_tasks=[${TASK}] \
        #                            method.lr=3e-4 framework.save_freq=100 framework.log_freq=10 \
         #               load=True framework.transitions_before_train=200 load_dir=${LOAD_DIR} load_step=${LOAD_STEP} framework.training_iterations=${STEP} \
         #           rlbench.demos=${N_DEMO} run_name=-${RUN}-Step${STEP}-Fill200-${N_DEMO}demo-3e4 rlbench.eval_tasks='${8_tasks}' framework.n_eval=4
done

done 
