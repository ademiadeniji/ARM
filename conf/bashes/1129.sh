# online, single emb: (encode either 8 or 16)
TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=3
RUN=10Var-Single-InfoNCE-QEncode8-Emb4x${DIM} 
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data contexts.sampler.k_dim=${Kdim} \
replay.buffers_per_batch=5 replay.batch_size=10 \
contexts.agent.single_embedding_replay=True dev.qnet_context_latent_size=8 


TAU=0.05
DIM=4
INFO_FREQ=2
Kdim=3
RUN=10Var-Single-InfoNCE-1Layer-Emb4x${DIM} 
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup'] rlbench.num_vars=10  \
run_name=${RUN} contexts.loss_mode='info' \
encoder.MODEL.OUT_DIM=${DIM} contexts.agent.tau=${TAU} contexts.agent.param_update_freq=${INFO_FREQ} \
rlbench.demo_path=/shared/mandi/all_rlbench_data contexts.sampler.k_dim=${Kdim} \
replay.buffers_per_batch=5 replay.batch_size=10 \
contexts.agent.single_embedding_replay=True dev.single_layer_context=True 
 

# skip hinge/infoNCE loss update every few steps 

FREQ=10
Query_ratio=0.3
K_dim=10
MAR=1e-2
E_LR=5e-4

RUN=10Var-Skip${FREQ}-Hinge-Emb8-2MLPQEncode8-1Layer
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=8  \
contexts.agent.query_ratio=${Query_ratio} method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.training_iterations=20000 contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
rlbench.demo_path=/shared/mandi/all_rlbench_data \
dev.single_layer_context=True dev.qnet_2_layer_context=True 

FREQ=100
Query_ratio=0.3
K_dim=10
MAR=1e-2
E_LR=5e-4

RUN=10Var-Skip${FREQ}-Hinge-Emb8-2MLPQEncode8-1Layer
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=8  \
contexts.agent.query_ratio=${Query_ratio} method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.training_iterations=20000 contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
rlbench.demo_path=/shared/mandi/all_rlbench_data \
dev.single_layer_context=True dev.qnet_2_layer_context=True 

FREQ=5
Query_ratio=0.3
K_dim=10
MAR=1e-2
E_LR=5e-4

RUN=10Var-Skip${FREQ}-Hinge-Emb8-2MLPQEncode8-1Layer
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=8  \
contexts.agent.query_ratio=${Query_ratio} method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.training_iterations=20000 contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
rlbench.demo_path=/shared/mandi/all_rlbench_data \
dev.single_layer_context=True dev.qnet_2_layer_context=True 



