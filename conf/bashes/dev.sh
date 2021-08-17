# taskset -c 20-40 python launch_multitask.py  tasks='${7_task_target}' replay.share_across_tasks=True run_name=OneBuffer-Batch63-lr5e4
# -- load agent and pruely generate attention 
RUN=7tasks-cup-lift-phone-rubbish-sauce-umbrella-wine/C2FARM-OneBuffer-Batch63-lr5e4/seed0
STEP=19900
TASK=reach_target
python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-55 dev.q_thres=0.55 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000;

python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-45 dev.q_thres=0.45 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000;

    python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-35 dev.q_thres=0.35 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000;

python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-65 dev.q_thres=0.65 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000;


RUN=7tasks-cup-lift-rubbish-sauce-target-umbrella-wine/C2FARM-Fresh-Batch63-lr5e4-Network3x16/seed0
STEP=26000
TASK=phone_on_base

    python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-35 dev.q_thres=0.35 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000 method.voxel_sizes=[16,16,16] method.bounds_offset=[0.15,0.15];

python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-45 dev.q_thres=0.45 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000 method.voxel_sizes=[16,16,16] method.bounds_offset=[0.15,0.15];

python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-55 dev.q_thres=0.55 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000 method.voxel_sizes=[16,16,16] method.bounds_offset=[0.15,0.15] ;

python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-65 dev.q_thres=0.65 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000 method.voxel_sizes=[16,16,16] method.bounds_offset=[0.15,0.15] ;

    python launch_multitask.py tasks=[$TASK]  resume=True run_name=VisOnly-NoTrain-Q_thres-75 dev.q_thres=0.75 \
    framework.training_iterations=100 framework.log_freq=10 resume_run=${RUN} resume_step=${STEP}  dev.save_freq=2000 method.voxel_sizes=[16,16,16] method.bounds_offset=[0.15,0.15] ;

## train multi-task
 python launch_multitask.py  tasks='${7_task_cup}' replay.share_across_tasks=True run_name=OneBuffer-Batch63-lr5e4 
  python launch_multitask.py  tasks='${7_task_cup}' replay.share_across_tasks=True run_name=OneBuffer-Batch126-lr5e4  replay.batch_size=18

# # Data gen:
cd .. 
python RLBench/tools/dataset_generator.py --save_path=/home/mandi/all_rlbench_data/ --image_size=128,128 --renderer=opengl --episodes_per_task=20 --variations=-1 --processes=10
