RUN=7tasks-cup-lift-phone-rubbish-sauce-target-umbrella/C2FARM-Fresh-Batch126-lr5e-4/seed0
STEP=24000
for DEMO in 1 10 0 
do

for BEFORE in 100 200
do
    python launch_multitask.py tasks=['stack_wine']  resume=True run_name=Resume-7Task-Batch64-lr1e4-${DEMO}Demo-${BEFORE}before \
    framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    resume_run=${RUN} resume_step=${STEP} \
    rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=1e-4; 

    python launch_multitask.py tasks=['stack_wine']  resume=True run_name=Resume-7Task-Batch64-lr2e4-${DEMO}Demo-${BEFORE}before \
    framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    resume_run=${RUN} resume_step=${STEP} \
    rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=2e-4; 

    python launch_multitask.py tasks=['stack_wine']  resume=True run_name=Resume-7Task-Batch64-lr3e4-${DEMO}Demo-${BEFORE}before \
    framework.wandb_logging=True framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    resume_run=${RUN} resume_step=${STEP} \
    rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=3e-4; 


    done
done
