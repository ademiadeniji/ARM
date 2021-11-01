

# ti1: remove prio replay, margin 5e-2 seems good for emb4 
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-QEncode8-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=8  contexts.pass_down_context=True 

   


    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-QEncode4-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=4 contexts.pass_down_context=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-NoQEncode-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10 \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True 

# ti5: try emb8 
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd8-Pretrain3k-QEncode8-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=2 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=8 contexts.pass_down_context=True \
     rlbench.demo_path=/shared/mandi/all_rlbench_data  

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd8-Pretrain3k-NoQEncode-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=2 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True \
     rlbench.demo_path=/shared/mandi/all_rlbench_data  


# amd2: No prio but both loss
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-QEncode8-NoPrio-BothLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 \
    contexts.agent.replay_update=True dev.qagent_update_context=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=8 contexts.pass_down_context=True \
     rlbench.demo_path=/shared/mandi/all_rlbench_data  

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-NoQEncode-NoPrio-BothLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 \
    contexts.agent.replay_update=True dev.qagent_update_context=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True \
     rlbench.demo_path=/shared/mandi/all_rlbench_data  

# txls1: more buffers: 12x10
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-NoQEncode-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=12 \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True \
     rlbench.demo_path=/shared/mandi/all_rlbench_data  

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-QEncode8-NoPrio-NoActionLoss
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=12 \
    contexts.agent.replay_update=True dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 \
    contexts.pretrain_replay_steps=3000 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=8 contexts.pass_down_context=True \
     rlbench.demo_path=/shared/mandi/all_rlbench_data  

    
# 1025
# ti1
    # debug: 100*emb_loss + no action loss
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    MAR=5e-2 
    RUN=10Var-DEBUG1k-NoActionLoss-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-NoPretrain-QEncode8 
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=8  contexts.pass_down_context=True 
    # debug -> pretrain then NO emb loss
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-DEBUG-NoHingeAfter-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-QEncode8 
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=8  contexts.pass_down_context=True \
    dev.qagent_use_emb_loss=False contexts.pretrain_replay_steps=3000 
    # enc 4
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-DEBUG-NoHingeAfter-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-QEncode4 
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=4  contexts.pass_down_context=True \
    dev.qagent_use_emb_loss=False contexts.pretrain_replay_steps=3000 


    # debug: 100*emb_loss + action loss
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    MAR=5e-2 
    RUN=10Var-DEBUG1k-WithAction-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-NoPretrain-QEncode8 
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    dev.qagent_update_context=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True dev.qnet_context_latent_size=8  contexts.pass_down_context=True 


    # no q encode
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6 
    MAR=5e-2 
    RUN=10Var-DEBUG1k-NoActionLoss-NoQEncode-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4 
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False  contexts.pass_down_context=True 

    # -> no emb loss after pre-train  
    RUN=10Var-DEBUG-NoHingeAfter-Mar${MAR}-Elr${E_LR}_K${K_dim}_Emd4-Pretrain3k-NoQEncode
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    dev.qagent_update_context=False \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False  \
    contexts.agent.embedding_size=1 framework.training_iterations=5000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False  contexts.pass_down_context=True \
    dev.qagent_use_emb_loss=False contexts.pretrain_replay_steps=3000 

    
