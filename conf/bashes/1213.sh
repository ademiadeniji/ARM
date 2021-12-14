DIM=16
HID=1024
RUN=10Var-dVAE-QEncode${DIM}-Hidden${HID}-NoArgmax
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
contexts.discrete_agent.one_hot=False dataset.num_steps_per_episode=1


DIM=16
HID=1024
RUN=10Var-Gumbel-3dim-discrete_before_hinge-QEncode${DIM}-Hidden${HID} 
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.defer_transforms=False dev.discrete=True contexts.loss_mode='gumbel' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=10 dev.encode_context_hidden=${HID} \
contexts.discrete_agent.latent_dim=3 \
dataset.num_steps_per_episode=1 encoder.DATA.NUM_FRAMES=1 \
contexts.discrete_agent.discrete_before_hinge=True 

DIM=16
HID=512
RUN=10Var-Gumbel-3dim-NoHinge-QEncode${DIM}-Hidden${HID} 
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.defer_transforms=False dev.discrete=True contexts.loss_mode='gumbel' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=10 dev.encode_context_hidden=${HID} \
contexts.discrete_agent.latent_dim=3  \
dataset.num_steps_per_episode=1 encoder.DATA.NUM_FRAMES=1 contexts.emb_weight=0

# hard update target embedder
FREQ=100
RUN=10Var-HardUpdate-Hinge-Freq${FREQ}-Emb64-QEncode16 
taskset -c $CPUS python launch_context.py run_name=${RUN} rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16 dev.encode_context=True dev.qnet_context_latent_size=16 \
framework.training_iterations=20000   contexts.agent.use_target_embedder=True contexts.agent.param_update_freq=${FREQ} \
dev.offline=True framework.wandb=False 
