# 0724: vary trainer-envrunner sync frequencies
for FREQ in 1 # 5 10 100 200
do 
taskset -c 0-15 python mt_launch.py rlbench.tasks='${4_task_A}' env_runner.receive=True rlbench.eval_tasks='${4_task_A}' run_name=-SyncFreq${FREQ}-Batch64-SetA-lr3e4-Voxel16x16 method.lr=3e-4 

done