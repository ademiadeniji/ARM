# STEP=22600
#RUN=7tasks-cup-lift-phone-rubbish-sauce-umbrella-wine/C2FARM-Fresh-Batch63-lr5e4-Network3x16/seed0 
RUN=/home/mandi/ARM/log/7tasks-lift-phone-rubbish-sauce-target-umbrella-wine/C2FARM-OneBuffer-Batch63-lr5e4/seed0
STEP=-1
for TASK in pick_up_cup phone_on_base pick_and_lift put_rubbish_in_bin stack_wine take_lid_off_saucepan take_umbrella_out_of_umbrella_stand
do
    python gen_video.py tasks=[$TASK] episodes=10 resume_step=${STEP} resume_run=${RUN} vid_name=7task-NoCup
done 
