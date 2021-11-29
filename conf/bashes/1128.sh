# infonce
TAU=0.1
DIM=4
for INFO_FREQ in 1 2 10 
do 
RUN=10Var-InfoNCE-Freeze2k-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=10000 dev.freeze_after_steps=3000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ}
done 
 
TAU=0.5
DIM=4
for INFO_FREQ in 1 2 10 
do 
RUN=10Var-InfoNCE-Freeze2k-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=10000 dev.freeze_after_steps=3000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ}
done 

TAU=0.5
DIM=4
for INFO_FREQ in 1 2 10 
do 
RUN=10Var-InfoNCE-Hidden64-Freeze-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=10000 dev.freeze_after_steps=3000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
contexts.agent.hidden_dim=64
done 

TAU=0.5
DIM=4
for INFO_FREQ in 1 2 10 
do 
RUN=10Var-InfoNCE-Hidden128-Freeze-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=10000 dev.freeze_after_steps=3000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
contexts.agent.hidden_dim=128
done 



TAU=0.01
INFO_FREQ=50
for DIM in 2 4 8 16 
do 
RUN=10Var-InfoNCE-Freeze2k-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=5000 dev.freeze_after_steps=2000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ}
done 

 
TAU=0.05
INFO_FREQ=50
for DIM in 2 4 8 16 
do 
RUN=10Var-InfoNCE-Freeze2k-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=5000 dev.freeze_after_steps=2000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ}
done 

TAU=0.05
INFO_FREQ=5
for DIM in 2 4 8 16 
do 
RUN=10Var-InfoNCE-Freeze2k-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=5000 dev.freeze_after_steps=2000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ}
done 


    