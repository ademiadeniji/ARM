# for TASK in stack_wine reach_target phone_on_base
# do

# # taskset -c 0-20 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-1Demo method.lr=3e-4 replay.batch_size=64 framework.training_iterations=25000 rlbench.demos=1

# # taskset -c 0-20 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-0Demo method.lr=3e-4 replay.batch_size=64 framework.training_iterations=25000 rlbench.demos=0

# # taskset -c 0-20 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-10Demo method.lr=3e-4 replay.batch_size=64 framework.training_iterations=25000 rlbench.demos=10


# # taskset -c 20-45 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-10Demo method.lr=3e-4 replay.batch_size=64 framework.replay_ratio=64 framework.training_iterations=5000 rlbench.demos=10

# # taskset -c 20-45 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-1Demo method.lr=3e-4 replay.batch_size=64 framework.replay_ratio=64 framework.training_iterations=5000 rlbench.demos=1

# # taskset -c 20-45 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-0Demo method.lr=3e-4 replay.batch_size=64 framework.replay_ratio=64 framework.training_iterations=5000 rlbench.demos=0

# taskset -c 20-45 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-10Demo-200before method.lr=3e-4 replay.batch_size=64 framework.replay_ratio=64 \
#     framework.training_iterations=5000 rlbench.demos=10 framework.transitions_before_train=200;

# taskset -c 20-45 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-1Demo-200before method.lr=3e-4 replay.batch_size=64 framework.replay_ratio=64 \
#     framework.training_iterations=5000 rlbench.demos=1 framework.transitions_before_train=200;

# taskset -c 20-45 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-0Demo-200before method.lr=3e-4 replay.batch_size=64 framework.replay_ratio=64 \
#     framework.training_iterations=5000 rlbench.demos=0 framework.transitions_before_train=200;




# done
 
## try solving new tasks 
DEM=10
BEF=200
for BEF in 100 50   
do
    for TASK in take_frame_off_hanger #hannoi_square #stack_cups # ut_plate_in_colored_dish_rack # setup_checkers wipe_desk #put_tray_in_oven

    do
        python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr5e4-${DEM}Demo-${BEF}before \
     method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 rlbench.demo_path=/home/mandi/all_rlbench_data wandb.group='unsolved'
done
done 
