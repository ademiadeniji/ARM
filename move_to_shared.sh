#for TASK in beat_the_buzz put_books_on_bookshelf block_pyramid put_bottle_in_fridg
#for TASK in change_channel  put_groceries_in_cupboard change_clock    put_item_in_drawer close_box   put_knife_in_knife_block
#for TASK in close_door                     put_knife_on_chopping_board close_drawer                   put_money_in_safe close_fridge                   put_plate_in_colored_dish_rack close_grill                    put_rubbish_in_bin close_jar                      put_shoes_in_box

#for TASK in close_laptop_lid               put_toilet_roll_on_stand close_microwave                put_tray_in_oven empty_container                put_umbrella_in_umbrella_stand empty_dishwasher               reach_and_drag get_ice_from_fridge            reach_target hang_frame_on_hanger           remove_cups hannoi_square                  scoop_with_spatula hit_ball_with_queue screw_nail

# for TASK in hockey  set_the_table insert_usb_in_computer         setup_checkers lamp_off    slide_block_to_target lamp_on       slide_cabinet_open light_bulb_in     slide_cabinet_open_and_place_cups light_bulb_out                 solve_puzzle meat_off_grill   stack_blocks meat_on_grill  stack_cups

# for TASK in memo.txt   stack_wine    move_hanger straighten_rope open_box sweep_to_dustpan  open_door  take_cup_out_from_cabinet   open_drawer     take_frame_off_hanger  open_fridge take_item_out_of_drawer     \
#     open_grill  take_lid_off_saucepan  open_jar take_money_out_safe open_microwave take_off_weighing_scales    open_oven  take_plate_off_colored_dish_rack     open_window  take_shoes_out_of_box    open_wine_bottle \
#     take_toilet_roll_off_stand phone_on_base  take_tray_out_of_oven pick_and_lift  
    
for TASK in take_umbrella_out_of_umbrella_stand pick_up_cup  take_usb_out_of_computer place_cups  \
    toilet_seat_down  place_hanger_on_rack toilet_seat_up place_shape_in_shape_sorter  turn_oven_on play_jenga turn_tap \
    plug_charger_in_power_supply   tv_off pour_from_cup_to_cup  tv_on press_switch   unplug_charger push_button       \
    water_plants push_buttons wipe_desk  put_all_groceries_in_cupboard
do
cp -r -n /home/mandi/all_rlbench_data/$TASK /shared/mandi/RLBENCH_DATA/ 
done
