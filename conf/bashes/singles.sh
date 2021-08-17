for TASK in pick_up_cup #stack_wine reach_target phone_on_base
do

# taskset -c 0-20 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-1Demo method.lr=3e-4 replay.batch_size=64 framework.training_iterations=25000 rlbench.demos=1

# taskset -c 0-20 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-0Demo method.lr=3e-4 replay.batch_size=64 framework.training_iterations=25000 rlbench.demos=0

# taskset -c 0-20 python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr3e4-10Demo method.lr=3e-4 replay.batch_size=64 framework.training_iterations=25000 rlbench.demos=10
#python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr5e4-10Demo-200before method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 framework.training_iterations=5000 rlbench.demos=10

#python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr5e4-1Demo-200before method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 framework.training_iterations=5000 rlbench.demos=1
for BEFORE in 100 50
do
python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr5e4-0Demo-${BEFORE}before method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 framework.training_iterations=5000 rlbench.demos=0 framework.transitions_before_train=${BEFORE}
done


done
