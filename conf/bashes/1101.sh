# joint training! bug should be fixed , try vary emb_weight
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd64-NoQEncode-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}

# change to pre-train:
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
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}
# change to pre-train + no action loss after
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd64-NoQEncode-Joint-EmbWeight_${WEI}-Pretrain2k-NoActionLoss 
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=False contexts.pretrain_replay_steps=3000 \
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
WEI=10
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd64-NoQEncode-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True \
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
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd64-QEncode32-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=True dev.qnet_context_latent_size=32 contexts.pass_down_context=True contexts.emb_weight=${WEI}


Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=10
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd64-QEncode32-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=True dev.qnet_context_latent_size=32 contexts.pass_down_context=True contexts.emb_weight=${WEI}

# pabti5: emb 16, sweep
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
WEI=1
for MAR in 5e-2 3e-2 1e-2 
do
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd16-QEncode8-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=True dev.qnet_context_latent_size=8 contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data
done 

Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
WEI=1
for MAR in 5e-2 3e-2 1e-2 
do
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd16-NoQEncode-Joint-EmbWeight_${WEI}
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=False contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data
done 

# rtxs1: add back prio + online switch -> try 5 var
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd64-NoQEncode-Joint-EmbWeight_${WEI}-Prio-Pretrain2k
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True   contexts.pretrain_replay_steps=2000 \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=True \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data

Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd64-NoQEncode-Joint-EmbWeight_${WEI}-Prio-Switch1-Pretrain2k
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
dev.qagent_update_context=True \
contexts.agent.query_ratio=${Query_ratio}  contexts.pretrain_replay_steps=2000 \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=True \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data framework.switch_online_tasks=1

# txl1: batch 128 
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=12
MAR=5e-2
WEI=1
RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd32-NoQEncode-Joint-EmbWeight_${WEI}-Prio
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
replay.batch_size=12 dev.qagent_update_context=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=True \
dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data  encoder.MODEL.OUT_DIM=8