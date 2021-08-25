## try solving new tasks 
# DEM=10
# BEF=200
# for TASK in 'open_door' 'take_off_weighing_scales' 'hit_ball_with_queue'  #'press_switch'  'open_box' 'light_bulb_out'
# do
# python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr5e4-${DEM}Demo-${BEF}before \
#      method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 \
# done

DEM=10
BEF=100
for BEF in 100 50 
do
for TASK in put_money_in_safe setup_checkers#put_money_in_safe # slide_cabinet_open_and_place_cups put_shoes_in_box  open_jar put_tray_in_oven 
do
python launch_multitask.py tasks=[${TASK}] run_name=Scratch-Batch64-lr5e4-${DEM}Demo-${BEF}before \
     method.lr=5e-4 replay.batch_size=64 framework.replay_ratio=64 rlbench.demo_path=/home/mandi/all_rlbench_data wandb.group='unsolved' framework.training_iterations=5000
done
done 

#cp -r -n /shared/mandi/all_rlbench_data/$TASK /home/mandi/all_rlbench_data/
# cp -r /home/mandi/all_rlbench_data/$TASK  /shared/mandi/all_rlbench_data/

# put_knife_on_chopping_board  straighten_rope hockey open_drawer beat_the_buzz 

# put_knife_in_knife_block put_all_groceries_in_cupboard scoop_with_spatula


## new solved ones:
# - open_door

