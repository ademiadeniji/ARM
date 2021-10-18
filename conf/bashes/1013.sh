# txl1: try multi task + multi var: try 3tasks, 60vars, 
 FREQ=100000
 Query_size=5
 E_LR=1e-4
 C_LR=1e-4 
 RUN=5VarEach-ReplayOnly-QEncodePass
 taskset -c 0-30 python launch_context.py run_name=${RUN} \
 tasks=['pick_up_cup','reach_target','pick_and_lift'] rlbench.num_vars=3 \
 replay.batch_size=12 replay.buffers_per_batch=10 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True \
 rlbench.demo_path=/shared/mandi/all_rlbench_data  \
  contexts.update_freq=${FREQ}  \
  contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
   method.emb_lr=${E_LR} contexts.sampler.batch_dim=10
    

# rtxs1: compare 3Var single and 3Var both batch
    Query_size=5
    FREQ=1
    E_LR=1e-4
    C_LR=1e-4 
    RUN=3Var-BothBatch-QEncodePass
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=32 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1,2] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ}  \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=10 \
    contexts.sampler.batch_dim=10
    
    multi task + multi var: try 3tasks,2vars, 
 FREQ=100000
 Query_size=5
 E_LR=1e-4
 C_LR=1e-4 
 RUN=2VarEach-ReplayOnly-QEncodePass
 python launch_context.py run_name=${RUN} \
 tasks=['pick_up_cup','reach_target','pick_and_lift'] rlbench.num_vars=2 \
 replay.batch_size=20 replay.buffers_per_batch=3 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True \
 rlbench.demo_path=/shared/mandi/all_rlbench_data  \
  contexts.update_freq=${FREQ}  \
  contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
   method.emb_lr=${E_LR} contexts.sampler.batch_dim=10


  #ti1: 3Task, dev 
 
    Query_size=5
    E_LR=1e-4
    C_LR=1e-4 
    RUN=2VarEach-ReplayOnly-NoHingeLoss
    python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','reach_target','pick_and_lift','reach_target'] rlbench.num_vars=2 \
    replay.batch_size=16 replay.buffers_per_batch=3 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True \
    rlbench.demo_path=/home/mandi/all_rlbench_data  \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} contexts.sampler.batch_dim=10  dev.qagent_use_emb_loss=False

    Query_size=5
    E_LR=1e-4
    C_LR=1e-4 
    RUN=2VarEach-OneContext
    python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','reach_target','pick_and_lift','reach_target'] rlbench.num_vars=2 \
    replay.batch_size=16 replay.buffers_per_batch=3 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True \
    rlbench.demo_path=/home/mandi/all_rlbench_data  \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} contexts.sampler.batch_dim=10  framework.wandb_logging=False



Query_size=5
    E_LR=1e-4
    C_LR=1e-4 
    RUN=10VarEach 
    python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','reach_target','pick_and_lift','push_button'] rlbench.num_vars=10 \
    replay.batch_size=12 replay.buffers_per_batch=5 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True \
    rlbench.demo_path=/home/mandi/all_rlbench_data  \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} contexts.sampler.batch_dim=10
    
      framework.wandb_logging=False

# debug: spreaded cpu 
Query_size=5
    E_LR=1e-4
    C_LR=1e-4 
    RUN=10VarEach 
    python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10 \
    replay.batch_size=12 replay.buffers_per_batch=5 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} contexts.sampler.batch_dim=10

# after limiting cpu, does it hurt to sample from a lot of buffers each time 
   Query_size=3
    E_LR=1e-4
    C_LR=1e-4 
    RUN=15VarEach 
    python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup','pick_and_lift','reach_target','push_button'] rlbench.num_vars=10 \
    replay.batch_size=4 replay.buffers_per_batch=8 replay.share_across_tasks=False \
    dev.encode_context=True contexts.pass_down_context=True \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} contexts.sampler.batch_dim=10 framework.wandb_logging=False