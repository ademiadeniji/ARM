for TASK in pick_up_cup phone_on_base pick_and_lift put_rubbish_in_bin reach_target stack_wine take_lid_off_saucepan take_umbrella_out_of_umbrella_stand

do

python mt_launch.py rlbench.tasks=[$TASK] run_name=-Batch64-lr3e4-Fill100-Demo0 rlbench.demos=0

done
