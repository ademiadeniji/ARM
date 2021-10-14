# txl1 2Var and 5 Var
    Query_size=6
    FREQ=100000
    E_LR=1e-4
    C_LR=1e-4 
    RUN=2Var-ReplayOnly-QEncodePass
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=16 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ} \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=16 # hack here to word around K dim 

    # see if no td error is required can be 
    Query_size=6
    FREQ=100000
    E_LR=1e-4
    C_LR=1e-4 
    RUN=2Var-ReplayOnly-OnlyHinge-QEncodePass 
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=16 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ} \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=False contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=16 # hack here to word around K dim 

    # 5var: 2x16 didn't throw error, try 3x16
    Query_size=6
    FREQ=100000
    E_LR=1e-4
    C_LR=1e-4 
    RUN=5Var-ReplayOnly-QEncodePass 
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=16 replay.buffers_per_batch=4 replay.share_across_tasks=False \
    dev.handpick=[0,1,2,3,4] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ} \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=16

# rtxs1 5Var both batch, try batch 16x2 -> breaks a lot
    Query_size=6
    FREQ=10
    E_LR=1e-4
    C_LR=1e-4 
    RUN=5Var-BothBatch-QEncodePass
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=16 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1,2,3,4] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ}  \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=16 contexts.sampler.batch_dim=5

# ti5: just 3 vars -> try without emb loss at all: dev.qagent_use_emb_loss=False
    Query_size=6
    FREQ=100000
    E_LR=1e-4
    C_LR=1e-4 
    RUN=3Var-ReplayOnly-QEncodePass
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=16 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1,2] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ} \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=16 # hack here to word around K dim 

    Query_size=6
    FREQ=100000
    E_LR=1e-4
    C_LR=1e-4 
    RUN=4Var-ReplayOnly-QEncodePass
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=16 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1,2,3] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ} \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=16

# ti1, 3 vars both batch, try batch 16x2
    Query_size=6
    FREQ=10
    E_LR=1e-4
    C_LR=1e-4 
    RUN=3Var-BothBatch-QEncodePass
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=16 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1,2] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ}  \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=16 contexts.sampler.batch_dim=3
    
    # replay only:
    Query_size=6
    FREQ=1000000 
    E_LR=1e-4
    C_LR=1e-4 
    RUN=3Var-ReplayOnly-QEncodePass
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=16 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1,2] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ}  \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=16 contexts.sampler.batch_dim=3

    # ti1
    # try loading back context-learned agent 
    LOAD=pick_up_cup-20var/3Var-BothBatch-QEncodePass-Replay_B16x2-Ctxt_B3_freq10_iter5_embed64/seed0 
    RUN=Load-3Var-BothBatchAgent
    STEP=4900

    LOAD=pick_up_cup-20var/3Var-ReplayOnly-QEncodePass-Replay_B16x2/seed0
    RUN=Load-3Var-ReplayOnlyAgent
    STEP=3400


    
    Query_size=6
    FREQ=1000000 
    E_LR=1e-4
    C_LR=1e-4 
    
    python launch_context.py tasks=['pick_up_cup'] dev.eval_only=True wandb.job_type=eval_only \
    resume=True resume_run=${LOAD} resume_step=${STEP} rlbench.demos=0     \
    replay.batch_size=16 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/home/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ}  \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20 framework.eval_envs=5  framework.training_iterations=50000
 
    