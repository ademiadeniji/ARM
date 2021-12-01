# dVAE pretrain
# pip install DALL-E
# wget https://cdn.openai.com/dall-e/encoder.pkl --no-check-certificate 
for DIM in 512 128 64 16
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
RUN=10Var-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden128
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.num_steps_per_episode=1 dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.

