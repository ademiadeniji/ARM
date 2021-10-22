# time's limited, all do batch 64
# ti1: embedding loss overfits. incrase k_dim for context part and decrese emb size 
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=0-16
    RUN=10VarEach-Ratio${RATIO}-Switch1-Margin005-Emb_lr${E_LR}-_K${K_dim}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    framework.switch_online_tasks=1  rlbench.num_vars=10  \
    rlbench.demo_path=/home/mandi/all_rlbench_data  

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    CPUS=16-32
    RUN=10VarEach-Ratio${RATIO}-Switch1-Margin005-Emb_lr${E_LR}-_K${K_dim}-Emb32
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    framework.switch_online_tasks=1  rlbench.num_vars=10  \
    rlbench.demo_path=/home/mandi/all_rlbench_data  contexts.agent.embedding_size=8

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=4
    RATIO=6
    CPUS=32-48
    RUN=10VarEach-Ratio${RATIO}-Switch1-Margin005-Emb_lr${E_LR}-_K${K_dim}-Emb32
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    framework.switch_online_tasks=1  rlbench.num_vars=10  \
    rlbench.demo_path=/home/mandi/all_rlbench_data  contexts.agent.embedding_size=8





# ti5 
    # -> increase emb_lr
    Query_ratio=0.3
    E_LR=5e-4 
    RATIO=6
    CPUS=0-24
    RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-Emb_lr${E_LR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    framework.switch_online_tasks=1  rlbench.num_vars=10  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  

    # no online switching -> no switch is better ?!
    Query_ratio=0.3
    E_LR=5e-4 
    RATIO=6
    CPUS=24-48
    RUN=10VarEach-Ratio${RATIO}-NoSwitch-Margin0-1-Emb_lr${E_LR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    rlbench.num_vars=10  rlbench.demo_path=/shared/mandi/all_rlbench_data  

    # no online switching -> no switch is better ?! -> 1) increase emb_lr 2) decrease emb_size
    Query_ratio=0.3
    E_LR=1e-3
    RATIO=6
    CPUS=24-48
    RUN=10VarEach-Ratio${RATIO}-NoSwitch-Margin0-1-Emb_lr${E_LR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    rlbench.num_vars=10  rlbench.demo_path=/shared/mandi/all_rlbench_data \
    contexts.agent.embedding_size=8

# rtxs1: bigger ratio  -> 12 doesn't work, try 9 + no switch
    Query_ratio=0.3
    E_LR=5e-4 
    RATIO=12
    CPUS=0-32
    RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-Emb_lr${E_LR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    framework.switch_online_tasks=1  rlbench.num_vars=10  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  
 

# txl1: 
    #  bigger ratio for larger batch 
    Query_ratio=0.3
    E_LR=5e-4 
    RATIO=12
    CPUS=0-32
    RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-Emb_lr${E_LR} 
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=12 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    framework.switch_online_tasks=1  rlbench.num_vars=10  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  
    #  bigger ratio for larger batch, smaller emb size
    Query_ratio=0.3
    E_LR=5e-4 
    RATIO=12
    CPUS=32-64
    RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-Emb_lr${E_LR}-Emb32
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
    replay.batch_size=10 replay.buffers_per_batch=12 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True   \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
    framework.replay_ratio=${RATIO} framework.log_all_vars=False \
    framework.switch_online_tasks=1  rlbench.num_vars=10  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  contexts.agent.embedding_size=8

 

# try qagent_update_context=False -> replay update is still true, this just means action loss doesn't affect
    # rtxs1:
        Query_ratio=0.3
        E_LR=5e-4 
        RATIO=6
        CPUS=0-32
        RUN=10VarEach-R${RATIO}-Switch1-Margin0-1-EmbLr${E_LR}-Emb8-NoActionLoss 
        taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
        replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
        dev.encode_context=True contexts.pass_down_context=True   \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
        method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
        framework.replay_ratio=${RATIO} framework.log_all_vars=False \
        framework.switch_online_tasks=1 rlbench.num_vars=10  \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  \
        dev.qagent_update_context=False contexts.agent.embedding_size=2
    
    # ti5: diff. emb lr and size
        Query_ratio=0.3
        E_LR=1e-3
        RATIO=6
        CPUS=0-24
        RUN=10VarEach-R${RATIO}-Switch1-Margin0-1-EmbLr${E_LR}-Emb8-NoActionLoss 
        taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
        replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
        dev.encode_context=True contexts.pass_down_context=True   \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
        method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
        framework.replay_ratio=${RATIO} framework.log_all_vars=False \
        framework.switch_online_tasks=1 rlbench.num_vars=10  \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  \
        dev.qagent_update_context=False contexts.agent.embedding_size=2

        Query_ratio=0.3
        E_LR=1e-4
        RATIO=6
        CPUS=24-48
        RUN=10VarEach-R${RATIO}-Switch1-Margin0-1-EmbLr${E_LR}-Emb8-NoActionLoss 
        taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
        replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
        dev.encode_context=True contexts.pass_down_context=True   \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
        method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
        framework.replay_ratio=${RATIO} framework.log_all_vars=False \
        framework.switch_online_tasks=1 rlbench.num_vars=10  \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  \
        dev.qagent_update_context=False contexts.agent.embedding_size=2

    # amd3 -> emb size 4
        Query_ratio=0.3
        RATIO=6
        for E_LR in 1e-4 5e-4 1e-3 
        do 
            RUN=10VarEach-R${RATIO}-Switch1-Margin0-1-EmbLr${E_LR}-Emb4-NoActionLoss 
            taskset -c $CPUS python launch_context.py run_name=${RUN} \
            tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
            replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
            dev.encode_context=True contexts.pass_down_context=True   \
            contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
            method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
            framework.replay_ratio=${RATIO} framework.log_all_vars=False \
            framework.switch_online_tasks=1 rlbench.num_vars=10  \
            rlbench.demo_path=/shared/mandi/all_rlbench_data  \
            dev.qagent_update_context=False contexts.agent.embedding_size=1 framework.training_iterations=5000
        done

    # txl1: bigger batch
        Query_ratio=0.3
        RATIO=12
        K_dim=16 
        for E_LR in 1e-4 5e-4 
        do 
            RUN=10VarEach-R${RATIO}-Switch1-Margin0-1-EmbLr${E_LR}-Emb4-NoActionLoss-K${K_dim}
            taskset -c $CPUS python launch_context.py run_name=${RUN} \
            tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
            replay.batch_size=10 replay.buffers_per_batch=12 replay.share_across_tasks=False \
            dev.encode_context=True contexts.pass_down_context=True   \
            contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
            method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
            framework.replay_ratio=${RATIO} framework.log_all_vars=False \
            framework.switch_online_tasks=1 rlbench.num_vars=10  \
            rlbench.demo_path=/shared/mandi/all_rlbench_data  \
            dev.qagent_update_context=False contexts.agent.embedding_size=1 framework.training_iterations=5000
        done

        Query_ratio=0.3
        RATIO=12
        K_dim=16 
        for E_LR in 1e-4 5e-4 
        do 
            RUN=10VarEach-R${RATIO}-Switch1-Margin001-EmbLr${E_LR}-Emb4-NoActionLoss-K${K_dim}
            taskset -c $CPUS python launch_context.py run_name=${RUN} \
            tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
            replay.batch_size=10 replay.buffers_per_batch=12 replay.share_across_tasks=False \
            dev.encode_context=True contexts.pass_down_context=True   \
            contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
            method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
            framework.replay_ratio=${RATIO} framework.log_all_vars=False \
            framework.switch_online_tasks=1 rlbench.num_vars=10  \
            rlbench.demo_path=/shared/mandi/all_rlbench_data  \
            dev.qagent_update_context=False contexts.agent.embedding_size=1 contexts.agent.margin=0.01 framework.training_iterations=5000
        done

    # amd2: No HingeLoss! 
        CPUS=120-160,180-255
        Query_ratio=0.3
        RATIO=6
        K_dim=10
        for E_LR in 1e-4 5e-4 
        do 
            RUN=10VarEach-R${RATIO}-Switch1-Margin001-EmbLr${E_LR}-Emb4-NoHinge-K${K_dim}
            taskset -c $CPUS python launch_context.py run_name=${RUN} \
            tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
            replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
            dev.encode_context=True contexts.pass_down_context=True   \
            contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
            method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
            framework.replay_ratio=${RATIO} framework.log_all_vars=False \
            framework.switch_online_tasks=1 rlbench.num_vars=10  \
            rlbench.demo_path=/shared/mandi/all_rlbench_data  \
            dev.qagent_update_context=False contexts.agent.embedding_size=1 contexts.agent.margin=0.01 \
            framework.training_iterations=5000 dev.qagent_use_emb_loss=False 
        done

# try pretrain with hinge loss 
    # ti1
        Query_ratio=0.3
        E_LR=5e-4
        K_dim=16 
        RATIO=3
        CPUS=0-16
        for MAR in 1e-2 3e-2 5e-2  
        do 
        RUN=10VarEach-Margin${MAR}-Emb_lr${E_LR}_K${K_dim}_Emd4-Pretrain2k
        taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
        replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
        dev.encode_context=True contexts.pass_down_context=True   \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
        method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
        framework.replay_ratio=${RATIO} framework.log_all_vars=False \
        framework.switch_online_tasks=1  rlbench.num_vars=10  \
        contexts.agent.embedding_size=1 \
        contexts.pretrain_replay_steps=2000 framework.training_iterations=3000 contexts.agent.margin=${MAR}
        done

        Query_ratio=0.3
        E_LR=1e-3
        K_dim=8 
        RATIO=3
        CPUS=16-32
        for MAR in 1e-2 3e-2 5e-2  
        do 
        RUN=10VarEach-Margin${MAR}-Emb_lr${E_LR}_K${K_dim}_Emd4-Pretrain2k
        taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
        replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
        dev.encode_context=True contexts.pass_down_context=True   \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
        method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
        framework.replay_ratio=${RATIO} framework.log_all_vars=False \
        framework.switch_online_tasks=1  rlbench.num_vars=10  \
        contexts.agent.embedding_size=1 \
        contexts.pretrain_replay_steps=2000 framework.training_iterations=3000 contexts.agent.margin=${MAR}
        done

        Query_ratio=0.3
        E_LR=5e-4
        K_dim=8
        RATIO=3
        CPUS=32-48
        for MAR in 1e-2 3e-2 5e-2  
        do 
        RUN=10VarEach-Margin${MAR}-Emb_lr${E_LR}_K${K_dim}_Emd8-Pretrain2k
        taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
        replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
        dev.encode_context=True contexts.pass_down_context=True   \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
        method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
        framework.replay_ratio=${RATIO} framework.log_all_vars=False \
        framework.switch_online_tasks=1  rlbench.num_vars=10  \
        contexts.agent.embedding_size=2 \
        contexts.pretrain_replay_steps=2000 framework.training_iterations=3000 contexts.agent.margin=${MAR}
        done
 
        # debug 
        python launch_context.py   \
        tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
        replay.batch_size=3 replay.buffers_per_batch=5 contexts.sampler.k_dim=2 \
        contexts.agent.replay_update=True  contexts.agent.embedding_size=2 \
        dev.offline=True framework.wandb_logging=False 

    # ti5: increase b_dim, keep pre-train 