# ti1: set small ratio  
Query_size=5
E_LR=1e-4
C_LR=1e-4 
RUN=10VarEach-SmallRatio3
taskset -c 20-48 python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target','push_button'] rlbench.num_vars=10  \
replay.batch_size=8 replay.buffers_per_batch=8 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size}  \
   method.emb_lr=${E_LR} contexts.sampler.batch_dim=10 framework.replay_ratio=3 framework.log_all_vars=False

Query_size=5
E_LR=1e-4
C_LR=1e-4 
RUN=10VarEach-SmallRatio6
taskset -c 0-20 python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target','push_button'] rlbench.num_vars=10  \
replay.batch_size=16 replay.buffers_per_batch=8 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size}  \
   method.emb_lr=${E_LR} contexts.sampler.batch_dim=10 framework.replay_ratio=6 framework.log_all_vars=False


# ti5: try switching only 2 online tasks out of the 4 total:

Query_size=3
E_LR=1e-4
C_LR=1e-4 
RUN=10VarEach-SmallRatio3-Switch2
python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target','push_button'] rlbench.num_vars=10  \
replay.batch_size=8 replay.buffers_per_batch=8 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size}  \
framework.switch_online_tasks=2 framework.wandb_logging=False rlbench.demos=1 rlbench.num_vars=2