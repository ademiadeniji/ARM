OUT=push_button
OUT=
OUT=pick_up_cup
RATIO=150
K=5
EPS=0.1
RUN=Anil-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_anil=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0] \
tasks.train_steps=50000 

# fine-tune freezes layer 0 and re-initialize layer1 
Task=push_button
RATIO=60 
STEP=8750 

Task=lamp_on
RATIO=30 
STEP=8500 

Task=take_lid_off_saucepan
STEP=8500
RATIO=60 

Task=pick_and_lift 
RATIO=30 
STEP=5750

Task=push_button
RATIO=60 
STEP=8750 

DEMO=0
RES=10Task-10var-Heldout-${Task}/Anil-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0 
RUN=FineTune-Anil-ResumeLayer1-Step${STEP}-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=${RATIO} rlbench.demos=${DEMO} framework.ckpt_eval=True \
replay.batch_size=60 replay.buffers_per_batch=1 \
framework.num_log_episodes=10 framework.eval_envs=5 \
replay.replay_size=100000 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 resume_freeze=[0] resume_path=/shared/mandi/arm_log 