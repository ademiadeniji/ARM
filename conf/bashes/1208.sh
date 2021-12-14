DIM=16
HID=4096
RUN=10Var-4step-dVAE-QEncode${DIM}-1Demo-1Layer-Hidden${HID}-OneHot
taskset -c $CPUS  python launch_context.py tasks=['pick_and_lift'] rlbench.num_vars=10 run_name=$RUN \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
dataset.num_steps_per_episode=4

DIM=16
HID=1024
RUN=10Var-dVAE-QEncode${DIM}-Hidden${HID}-NoArgmax
taskset -c $CPUS  python launch_context.py rlbench.num_vars=10 run_name=$RUN \
dataset.defer_transforms=True dev.discrete=True contexts.loss_mode='dvae' \
dev.qnet_context_latent_size=${DIM} framework.training_iterations=20000 \
contexts.sampler.k_dim=1  dev.single_layer_context=True dev.encode_context_hidden=${HID} \
contexts.discrete_agent.one_hot=False 


# eval zero shot on unseen vars 
RUN=ZeroShot-dVAE-EvalOnly 
taskset -c $CPUS  python launch_context.py run_name=$RUN \
rlbench.demos=0 dev.eval_only=True resume=True \
resume_run='pick_up_cup-20var/10Var-dVAE-QEncode16-1Demo-1Layer-Hidden512-OneHot-Replay_B10x6-Q0.3/seed0' \
resume_step=9900 dev.discrete=True contexts.loss_mode='dvae' dataset.defer_transforms=True \
contexts.sampler.k_dim=1  dataset.num_steps_per_episode=1 \
dev.encode_context_hidden=512 dev.qnet_context_latent_size=16 framework.training_iterations=50000


# discretise hinge 

E_LR=5e-4
RUN=10Var-Discretise-Hinge-Emb64-QEncode8 
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16 dev.encode_context=True dev.qnet_context_latent_size=8 \
framework.training_iterations=20000   dev.discretise=True 
 
E_LR=5e-4
RUN=10Var-Discretise-Hinge-Emb64-QEncode16 
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16 dev.encode_context=True dev.qnet_context_latent_size=16 \
framework.training_iterations=20000   dev.discretise=True 

E_LR=5e-4
RUN=10Var-Discretise-InfoNCE-Emb64-QEncode16 
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16 dev.encode_context=True dev.qnet_context_latent_size=16 \
framework.training_iterations=20000   dev.discretise=True contexts.loss_mode='info'

E_LR=5e-4
RUN=10Var-Discretise-InfoNCE-1Sample-Emb64-QEncode16 
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16 dev.encode_context=True dev.qnet_context_latent_size=16 \
framework.training_iterations=20000   dev.discretise=True contexts.loss_mode='info' dev.single_layer_context=True 

E_LR=0
RUN=10Var-Discretise-0loss-InfoNCE-1Sample-Emb64-QEncode16 
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16 dev.encode_context=True dev.qnet_context_latent_size=16 \
framework.training_iterations=20000   dev.discretise=True contexts.loss_mode='info' dev.single_layer_context=True 



E_LR=1e-4
RUN=10Var-Discretise-Hinge-Emb64-QEncode8-Elr${E_LR}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16 dev.encode_context=True dev.qnet_context_latent_size=8 \
framework.training_iterations=20000   dev.discretise=True method.emb_lr=${E_LR}

E_LR=1e-4
RUN=10Var-Discretise-Hinge-Emb64-QEncode8-Elr${E_LR}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10 \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16 dev.encode_context=True dev.qnet_context_latent_size=16 \
framework.training_iterations=20000   dev.discretise=True method.emb_lr=${E_LR}