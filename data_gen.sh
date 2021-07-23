# *8* single-camera tasks from C2FARM:
# python ../RLBench/tools/dataset_generator.py --save_path=/home/mandi/ARM/data/ \
#     --tasks=pick_up_cup,phone_on_base,pick_and_lift,put_rubbish_in_bin,reach_target,stack_wine,take_lid_off_saucepan,take_umbrella_out_of_umbrella_stand \
#     --image_size=128,128 --renderer=opengl --episodes_per_task=10 --variations=1 --processes=1
python ../RLBench/tools/dataset_generator.py --save_path=/home/mandi/ARM/data/ \
    --tasks=reach_target,stack_wine,take_lid_off_saucepan,take_umbrella_out_of_umbrella_stand \
    --image_size=128,128 --renderer=opengl --episodes_per_task=10 --variations=1 --processes=1

# *4* multi-camera tasks from C2FARM:
python ../RLBench/tools/dataset_generator.py --save_path=/home/mandi/ARM/data/ \
    --tasks=meat_off_grill,put_groceries_in_cupboard,take_money_out_safe,unplug_charger \
    --image_size=128,128 --renderer=opengl --episodes_per_task=10 --variations=1 --processes=1