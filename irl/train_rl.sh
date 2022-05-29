# rl train:
python launch.py rew.scale_logits=True run_name=TunedItr0-Aug3 \
    rew.model='Iteration0-5x20Batch' rew.step=450 rew.save_data=True rew.use_aug=True rew.aug_avg=3 

MODEL=Iteration0-3x40Batch-SmoothSuccess
STEP=400

MODEL=Iteration0-3x50Batch-FailOnly
STEP=900

MODEL=FailOnly-1x100
STEP=1000
RUN=TunedItr0-FailOnly-1x100

MODEL=LatestData-FailOnly-2x150Batch
STEP=300
RUN=Button-LatestTuned-Scale10
python launch.py rew.scale_const=10 run_name=$RUN \
    rew.model=$MODEL rew.step=$STEP rew.use_aug=True rew.save_data=False  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data

# saucepan!
STEP=400
MODEL=Saucepan-DemoOnly

STEP=300
MODEL=Saucepan-Demo-Fail-3x50Batch

MODEL=Saucepan-3Levels-2x50Batch
STEP=300
RUN=3Levels-3x50Batch
python launch.py rew.scale_logits=True run_name=$RUN \
    rew.model=$MODEL rew.step=$STEP rew.use_aug=True \
    rlbench.demo_path=/shared/mandi/all_rlbench_data \
    tasks.all_tasks=[take_lid_off_saucepan] \
    resume=True resume_path=/home/mandi/ARM/irl/log/ \
    resume_run=take_lid_off_saucepan-1var/Demo-Fail-Replay_B60x1/seed2 resume_step=5000

# launch but finetune
RUN=ResumeTaskReward-Ratio30
RESUME=push_button-18var/R3MReward-Replay_B60x1/seed10
STEP=9999
python launch.py run_name=$RUN \
    rew.save_data=False resume=True rew.task_reward=True framework.replay_ratio=30  \
    resume_run=$RESUME resume_step=$STEP \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    resume_path=/shared/mandi/irl_arm_log/ \

# task reward saucepan
python launch.py rew.scale_logits=True run_name=TaskReward \
rew.task_reward=True rlbench.demo_path=/shared/mandi/all_rlbench_data \
tasks.all_tasks=[take_lid_off_saucepan] resume=True \
resume_path=/home/mandi/ARM/irl/log/ rew.task_name=take_lid_off_saucepan \
resume_step=3000 resume_run=take_lid_off_saucepan-1var/3Levels-3x50Batch-Replay_B60x1/seed6



# R3M reward flipped!
python launch.py run_name=R3MReward rew.model='r3m' rew.prompts=['reach touch'] \
    rew.use_r3m=True framework.wandb=False 

# R3M reward
python launch.py run_name=R3M-Tuned-Reward rew.model='r3m' rew.prompts=['take lid off saucepan'] \
    rew.use_r3m=True  tasks.all_tasks=[take_lid_off_saucepan] 

RUN=Resume-TunedR3M-Ratio30
RESUME=push_button-18var/R3M-Tuned-Reward-Replay_B60x1/seed2
STEP=6800

RUN=Resume-TunedR3M-Ratio$RATIO
RESUME=take_lid_off_saucepan-1var/R3M-Tuned-Reward-Replay_B60x1/seed1 
STEP=9999

RUN=Resume-10TaskR3M
RESUME=10Task-10var/R3M-Tuned-Reward-Replay_B10x6/seed0 
STEP=15000

RUN=Resume-10TaskSparse
RESUME=10Task-10var/SparsePreTrain-Replay_B10x6/seed0
STEP=15000
RATIO=60
python launch.py run_name=$RUN \
    rew.save_data=False resume=True rew.task_reward=True framework.replay_ratio=$RATIO  \
    resume_run=$RESUME resume_step=$STEP tasks.all_tasks=[put_rubbish_in_bin] \
    resume_path=/shared/mandi/arm_log/ 

    rlbench.demo_path=/shared/mandi/arm_log  \
    \

python launch.py run_name=R3M-Tuned-Reward rew.model='r3m' rew.use_r3m=True \
 tasks.all_tasks=[put_rubbish_in_bin] 
# multi task
python launch.py run_name=160kR3M-Tuned-Reward rew.model='r3m' rew.use_r3m=True \
 tasks.all_tasks=[take_lid_off_saucepan,push_button,reach_target,pick_up_cup,phone_on_base,lamp_on,put_umbrella_in_umbrella_stand,put_rubbish_in_bin,stack_wine,take_usb_out_of_computer] \
 tasks.num_vars=1 tasks.train_steps=100_000 replay.batch_size=10 replay.num_tasks_per_batch=6 framework.replay_ratio=60 framework.replay_ratio=60 \
 framework.transitions_before_train=10 rlbench.demo_path=/shared/mandi/all_rlbench_data 
# compare against task rew:
python launch.py run_name=SparsePreTrain rew.task_reward=True \
 tasks.all_tasks=[take_lid_off_saucepan,push_button,reach_target,pick_up_cup,phone_on_base,lamp_on,put_umbrella_in_umbrella_stand,put_rubbish_in_bin,stack_wine,take_usb_out_of_computer] \
 tasks.num_vars=1 tasks.train_steps=100_000 framework.replay_ratio=60 replay.batch_size=10 rlbench.demo_path=/shared/mandi/all_rlbench_data framework.transitions_before_train=10

# test on pick_and_lift