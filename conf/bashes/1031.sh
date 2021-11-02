# hw2: MT 4 task

# rtxl1
export CUDA_VISIBLE_DEVICES=0,1,2
CPUS=0-16
RATIO=64
RUN=Hw2-OneBuffer
taskset -c $CPUS python launch_context.py run_name=${RUN} mt_only=True \
tasks=['push_button','pick_and_lift','reach_target','take_lid_off_saucepan'] rlbench.num_vars=1 \
framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
replay.share_across_tasks=True replay.batch_size=128 rlbench.demos=3 rlbench.demo_path=/shared/mandi/all_rlbench_data

export CUDA_VISIBLE_DEVICES=2,3,4
CPUS=16-32
RATIO=64
RUN=Hw2-4Buffer-Uniform
taskset -c $CPUS python launch_context.py run_name=${RUN} mt_only=True \
tasks=['push_button','pick_and_lift','reach_target','take_lid_off_saucepan'] rlbench.num_vars=1 \
framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
replay.share_across_tasks=False replay.buffers_per_batch=4  \
replay.batch_size=32 rlbench.demos=3 rlbench.demo_path=/shared/mandi/all_rlbench_data

export CUDA_VISIBLE_DEVICES=4,5,6
CPUS=32-48
RATIO=64
RUN=Hw2-4Buffer-Prio 
taskset -c $CPUS python launch_context.py run_name=${RUN} mt_only=True \
tasks=['push_button','pick_and_lift','reach_target','take_lid_off_saucepan'] rlbench.num_vars=1 \
framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
replay.share_across_tasks=False replay.buffers_per_batch=2  \
replay.batch_size=64 rlbench.demos=3 rlbench.demo_path=/shared/mandi/all_rlbench_data

export CUDA_VISIBLE_DEVICES=7,6,5
CPUS=48-64
RATIO=64
RUN=Hw2-4Buffer-NoPrio 
taskset -c $CPUS python launch_context.py run_name=${RUN} mt_only=True \
tasks=['push_button','pick_and_lift','reach_target','take_lid_off_saucepan'] rlbench.num_vars=1 \
framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
replay.share_across_tasks=False replay.buffers_per_batch=2 replay.update_buffer_prio=False \
replay.batch_size=64 rlbench.demos=3 rlbench.demo_path=/shared/mandi/all_rlbench_data

# rtxs1: bsize64
CPUS=48-64
RATIO=6
RUN=Hw2-6Buffer-NoPrio 
taskset -c $CPUS python launch_context.py run_name=${RUN} mt_only=True \
tasks=['push_button','pick_and_lift','reach_target','take_lid_off_saucepan','lamp_on','put_rubbish_in_bin'] rlbench.num_vars=1 \
framework.replay_ratio=${RATIO} framework.training_iterations=50000 \
replay.share_across_tasks=False replay.buffers_per_batch=4 replay.update_buffer_prio=False \
replay.batch_size=16 rlbench.demos=3 rlbench.demo_path=/shared/mandi/all_rlbench_data

CPUS=32-48
RATIO=6
RUN=Hw2-6Buffer-Prio
taskset -c $CPUS python launch_context.py run_name=${RUN} mt_only=True \
tasks=['push_button','pick_and_lift','reach_target','take_lid_off_saucepan','lamp_on','put_rubbish_in_bin'] rlbench.num_vars=1 \
framework.replay_ratio=${RATIO} framework.training_iterations=50000 \
replay.share_across_tasks=False replay.buffers_per_batch=4 replay.update_buffer_prio=True \
replay.batch_size=16 rlbench.demos=3 rlbench.demo_path=/shared/mandi/all_rlbench_data

# continue on pick up cup
DEM=0
RATIO=6
RUN=Hw2-Resume-4Task-Demo${DEM}
taskset -c $CPUS python launch_context.py run_name=${RUN} mt_only=True \
tasks=['pick_up_cup'] rlbench.num_vars=1 \
framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
replay.share_across_tasks=True replay.batch_size=128 rlbench.demos=${DEM} \
rlbench.demo_path=/shared/mandi/all_rlbench_data \
resume_run=4Task-4var/Hw2-4Buffer-Prio-Replay_B64x2-Q0.3-NoContext/seed2 resume_step=19900


# ti1: debug why after pretrain + No Action loss, the emb acc still goes down 
# disable online training, no pretrain -> if emb accuracy goes down, 
    # offline w/ or w/o action loss 
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Offline-NoQEncode-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=0 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True dev.offline=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=16-32
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Offline-NoQEncode-NoPrio-WithActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    contexts.agent.replay_update=True dev.qagent_update_context=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=0 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True dev.offline=True 

    # detach AND make a copy 
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Offline-Copy-NoQEncode-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=0 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True dev.offline=True 

    # detach AND make a copy 
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Offline-Copy-NoQEncode-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=0 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True dev.offline=True 

    # detach, clone, step twice!
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Offline-Copy-StepTwice-NoQEncode-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=0 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True dev.offline=True 




# correct way to set emb size:
encoder.MODEL.OUT_DIM=2 
# detach AND make a copy 
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    MAR=1e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_CorrectEmd8-Offline-Copy-NoQEncode-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    encoder.MODEL.OUT_DIM=2 \
    contexts.pretrain_replay_steps=1000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True dev.offline=True 



