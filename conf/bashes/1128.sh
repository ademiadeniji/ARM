# infonce
# smaller k_dim, bigger b_dim, bigger hidden_dim 



# online, single emb:
TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=3
RUN=10Var-Single-InfoNCE-Emb4x${DIM} 
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data contexts.sampler.k_dim=${Kdim} \
replay.buffers_per_batch=10 \
contexts.agent.single_embedding_replay=True 

TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=3
RUN=10Var-Single-InfoNCE-Emb4x${DIM} 
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data contexts.sampler.k_dim=${Kdim} \
replay.buffers_per_batch=5 replay.batch_size=20 \
contexts.agent.single_embedding_replay=True 

TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=3
RUN=10Var-Single-InfoNCE-Emb4x${DIM} 
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data contexts.sampler.k_dim=${Kdim} \
replay.buffers_per_batch=10 replay.batch_size=5 \
contexts.agent.single_embedding_replay=True 


TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=2
RUN=20Var-InfoNCE-Freeze2k-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}-K_dim${Kdim}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=20  \
run_name=${RUN} contexts.pretrain_replay_steps=20000 dev.freeze_after_steps=5000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data contexts.sampler.k_dim=${Kdim} \
replay.buffers_per_batch=15

TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=3
RUN=10Var-InfoNCE-Freeze-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}-K_dim${Kdim}-Pred128
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=20000 dev.freeze_after_steps=5000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data \
contexts.sampler.k_dim=${Kdim} replay.buffers_per_batch=10 \
contexts.agent.hidden_dim=128


TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=6
RUN=10Var-InfoNCE-Freeze2k-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}-K_dim${Kdim}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=20000 dev.freeze_after_steps=5000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data contexts.sampler.k_dim=${Kdim} replay.buffers_per_batch=10

TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=3
RUN=10Var-InfoNCE-Freeze-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}-K_dim${Kdim}-E_LR1e-4
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=20000 dev.freeze_after_steps=5000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data contexts.sampler.k_dim=${Kdim} \
replay.buffers_per_batch=10 method.emb_lr=1e-4



TAU=0.05
DIM=4
for INFO_FREQ in 2
do 
RUN=10Var-InfoNCE-Freeze2k-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=10000 dev.freeze_after_steps=3000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data
done 

TAU=0.05
DIM=4
for INFO_FREQ in 5 10 
do 
RUN=10Var-InfoNCE-Hidden64-Freeze-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=10000 dev.freeze_after_steps=3000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
contexts.agent.hidden_dim=64 rlbench.demo_path=/shared/mandi/all_rlbench_data
done 

TAU=0.05
DIM=4
INFO_FREQ=2
E_LR=1e-4
RUN=10Var-InfoNCE-Freeze-Classify-Emb4x${DIM}-Tau${TAU}-Soft${INFO_FREQ}-E_LR${E_LR}
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.pretrain_replay_steps=10000 dev.freeze_after_steps=5000 \
dev.classify=True dev.offline=True framework.log_freq=50 contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} \
rlbench.demo_path=/home/mandi/front_rlbench_data \
contexts.agent.param_update_freq=${INFO_FREQ} method.emb_lr=${E_LR} 



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


    