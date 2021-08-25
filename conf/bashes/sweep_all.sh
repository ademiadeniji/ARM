DEM=10 
for BEF in 200  
do
python sweep_all_tasks.py run_name=Sweep-Batch64-lr5e4-${DEM}Demo-${BEF}before \
     method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 rlbench.demo_path=/home/mandi/all_rlbench_data \
        wandb.group='sweep' framework.training_iterations=5000 framework.transitions_before_train=200

done 