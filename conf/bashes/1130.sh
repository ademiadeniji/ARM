# dVAE pretrain
# pip install DALL-E
# wget https://cdn.openai.com/dall-e/encoder.pkl --no-check-certificate 
for DIM in 64 16
do
RUN=10Var-dVAE-Classify-Hidden${DIM}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True \
dev.offline=True contexts.loss_mode='dvae' \
contexts.pretrain_replay_steps=10000 dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=6 dev.classify=True framework.log_freq=50 
done

DIM=8
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True 
 

DIM=8
HID=32
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} rlbench.demo_path=/shared/mandi/all_rlbench_data

DIM=16
HID=128
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} rlbench.demo_path=/shared/mandi/all_rlbench_data

# txl1
DIM=32
HID=-1
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} rlbench.demo_path=/shared/mandi/all_rlbench_data


DIM=16
HID=-1
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} rlbench.demo_path=/shared/mandi/all_rlbench_data

# normalize
DIM=16
HID=-1
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}-Normalize
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID}

DIM=16
HID=64
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}-Normalize
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID}

DIM=8
HID=512
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}-Norm1
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID}


DIM=16
HID=1024
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}-OneHot
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} contexts.discrete_agent.one_hot=True

DIM=16
HID=1024
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}-OneHot
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup','pick_and_lift'] rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000  \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} contexts.discrete_agent.one_hot=True

DIM=16
HID=1024
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}-OneHot
taskset -c $CPUS  python launch_context.py tasks=['pick_up_cup','reach_target'] rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} contexts.discrete_agent.one_hot=True



# take mean from 3 embs 
DIM=16
HID=128
RUN=10Var-dVAE-QEncode${DIM}-Mean6Emb-1Layer-Hidden${HID}-Norm1
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
dev.single_layer_context=False dev.encode_context_hidden=${HID} \
contexts.sampler.k_dim=6 dev.offline=True 

DIM=16
RUN=10Var-dVAE-Classify-Range1-Single-Hidden${DIM}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True \
dev.offline=True contexts.loss_mode='dvae' \
contexts.pretrain_replay_steps=30000 dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=3 dev.classify=True framework.log_freq=50 


# Skip!! 
FREQ=50
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
dev.single_layer_context=True dev.qnet_2_layer_context=True dev.replay_update_freq=${FREQ}
 