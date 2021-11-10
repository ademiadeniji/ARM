# ti1: try just 1 task
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=20Var-Emd16-NoQEncode-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=20  \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}

Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=20Var-Emd8-NoQEncode-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=20  \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}


Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=20Var-Emd16-NoQEncode-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_and_lift'] rlbench.num_vars=20  \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}

# ti5: try fewer vars

Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=3Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd8-NoQEncode-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=3  \
dev.qagent_update_context=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=2 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}

Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=3Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd16-NoQEncode-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=3  \
dev.qagent_update_context=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} encoder.MODEL.OUT_DIM=4 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI} replay.update_buffer_prio=True 

# ti1:
# change to pre-train:
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd32-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=8 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}

Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd64-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=16 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}


Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=20Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd32-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_and_lift'] rlbench.num_vars=20  \
dev.qagent_update_context=True contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=8 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}

Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=20Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd16-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_and_lift'] rlbench.num_vars=20  \
dev.qagent_update_context=True contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=4 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}

# pabti5

Query_ratio=0.3
E_LR=1e-3
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=20Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd32-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_and_lift'] rlbench.num_vars=20  \
dev.qagent_update_context=True contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=8 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI} rlbench.demo_path=/shared/mandi/all_rlbench_data

Query_ratio=0.3
E_LR=1e-3
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=20Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd16-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_and_lift'] rlbench.num_vars=20  \
dev.qagent_update_context=True contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=4 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data

# rtxs1
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=1e-3
WEI=1
RUN=20Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd16-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_and_lift'] rlbench.num_vars=20  \
dev.qagent_update_context=True contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=4 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data
 
 # txl1: big batch

Query_ratio=0.3
E_LR=5e-4
K_dim=32
RATIO=12
MAR=5e-2 
WEI=1
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd32-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio} replay.batch_size=12 \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False encoder.MODEL.OUT_DIM=8 \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}