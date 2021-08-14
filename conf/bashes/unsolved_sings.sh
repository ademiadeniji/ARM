## try solving new tasks 
# DEM=10
# BEF=200
# for TASK in 'open_door' 'take_off_weighing_scales' 'hit_ball_with_queue'  #'press_switch'  'open_box' 'light_bulb_out'
# do
# python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr5e4-${DEM}Demo-${BEF}before \
#      method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 \
# done

DEM=10
BEF=200
for TASK in 'press_switch'  #'open_box' 'light_bulb_out'
do
python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr5e4-${DEM}Demo-${BEF}before \
     method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 rlbench.demo_path=/home/mandi/all_rlbench_data wandb.group='unsolved'
done

cp -r -n /shared/mandi/all_rlbench_data/$TASK /home/mandi/all_rlbench_data/