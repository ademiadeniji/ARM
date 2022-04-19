# cup: amd2 need retrain
RES=10Task-10var-Heldout-pick_up_cup/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0

OUT=pick_up_cup
RATIO=150
K=5
EPS=0.1
RUN=Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_reptile=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0.084] \
tasks.train_steps=8000 ;

RES=10Task-10var-Heldout-pick_up_cup/Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}-Replay_B15x10/seed0
for STEP in 500 1000 2500 7500 
do
RUN=EvalReptile-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done 

# button amd1 
RES=10Task-10var-Heldout-push_button/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0
STEP=7500  
RUN=EvalReptile-Step${STEP}-NoContext
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=${RATIO} framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 

OUT=push_button
RATIO=150
K=5
EPS=0.1
RUN=Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_reptile=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0.098] \
tasks.train_steps=1001 ;
RES=10Task-10var-Heldout-push_button/Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}Replay_B15x10/seed0
for STEP in 500 1000 
do
RUN=EvalReptile-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done 

# lift amd1 
OUT=pick_and_lift
RATIO=150
K=5
EPS=0.1
RUN=Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_reptile=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0.084] \
tasks.train_steps=8000 ;

RES=10Task-10var-Heldout-${OUT}/Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}-Replay_B15x10/seed0
for STEP in 500 2500 7500 
do
RUN=EvalReptile-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done 

# ti5 
OUT=lamp_on
RATIO=150
K=5
EPS=0.1
RUN=Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_reptile=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0.084] \
tasks.train_steps=8000 ;

RES=10Task-10var-Heldout-${OUT}/Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}Replay_B15x10/seed0
for STEP in 500 2500 7500 
do
RUN=EvalReptile-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done 

# ti5
OUT=take_lid_off_saucepan
RES=10Task-10var-Heldout-${OUT}/Reptile-Kstep5-Eps0.1-1Var-1Cam-10Buffer-Ratio150-Replay_B15x10/seed0

for STEP in 10000 12500 15000 17500 20000  
do
RUN=EvalReptile-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 resume_path=/shared/mandi/arm_log 
done 
# amd2
OUT=take_lid_off_saucepan
RATIO=150
K=5
EPS=0.1
RUN=Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}
taskset -c $CPUS python launch_context.py mt_only=True \
rlbench.num_vars=1 run_name=${RUN} \
tasks.heldout=${OUT} framework.replay_ratio=$RATIO \
replay.buffers_per_batch=10 rlbench.demos=5 framework.save_freq=250 \
replay.batch_size=15 framework.log_freq=250  \
rlbench.num_vars=1 dev.use_reptile=True dev.reptile_k=${K} dev.reptile_eps=[${EPS},0.084] \
tasks.train_steps=8000 ;

RES=10Task-10var-Heldout-${OUT}/Retrain-Reptile-Kstep${K}-Eps${EPS}-1Var-1Cam-10Buffer-Ratio${RATIO}-Replay_B15x10/seed0
for STEP in 500 1000 2500 7500 
do
RUN=EvalReptile-Step${STEP}-NoContext-1Buffer 
taskset -c $CPUS python launch_context.py run_name=$RUN mt_only=True \
tasks.heldout=''  \
framework.replay_ratio=1 rlbench.demos=0 framework.ckpt_eval=True \
replay.batch_size=1 replay.buffers_per_batch=1 \
framework.eval_envs=1 \
replay.replay_size=100 resume=True resume_run=${RES} resume_step=${STEP} \
rlbench.num_vars=1 tasks.train_steps=1 \
framework.transitions_before_train=10 rlbench.demos=1 framework.train_envs=0 \
framework.num_log_episodes=30 
done 