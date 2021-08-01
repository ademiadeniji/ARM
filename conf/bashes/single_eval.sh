for TASK in phone_on_base stack_wine take_lid_off_saucepan take_umbrella_out_of_umbrella_stand

do
python load_inspect.py load=true load_dir=/home/mandi/ARM/log/7tasks-cup-lift-phone-rubbish-sauce-target-umbrella/C2FARM-Hard-Batch126-Voxel16x16-5e4/seed2/weights run_name=7TaskWine rlbench.tasks=[${TASK}]

done
