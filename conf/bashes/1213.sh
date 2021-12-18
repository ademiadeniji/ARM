DIM=16
HID=1024
RUN=10Var-dVAE-QEncode${DIM}-Hidden${HID}-NoArgmax
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
contexts.discrete_agent.one_hot=False dataset.num_steps_per_episode=1


DIM=8
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
FREQ=2000
RUN=10Var-NewV1-HardUpdate-Hinge-Freq${FREQ}-Emb16-QEncode8
taskset -c $CPUS python launch_context.py run_name=${RUN} rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=8 \
framework.training_iterations=20000   contexts.agent.use_target_embedder=True contexts.agent.param_update_freq=${FREQ} 

DIM=16
HID=1024
RUN=10Var-Stack-2frame-dVAE-QEncode${DIM}-Hidden${HID}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=2

DIM=16
HID=1024
OUT=2048
RUN=10Var-3frameConv3d-dVAE-QEncode${DIM}-Hidden${HID}-Ker311-Feat${OUT}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN tasks=['pick_and_lift'] \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=3 cdev.use_conv=True cdev.conv_out=${OUT} 

DIM=16
HID=1024
OUT=4096
RUN=10Var-3frameConv3d-dVAE-QEncode${DIM}-Hidden${HID}-Ker322-Feat${OUT}
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN tasks=['pick_and_lift'] \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=3 cdev.use_conv=True cdev.conv_out=${OUT} cdev.conv_kernel=[3,2,2]