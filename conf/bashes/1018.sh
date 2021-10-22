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

# 16x8
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

# 1019: no match
Query_ratio=0.3
E_LR=1e-4
C_LR=1e-4 
RATIO=3
RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-NoMatch
taskset -c 0-20 python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target','push_button'] rlbench.num_vars=10  \
replay.batch_size=8 replay.buffers_per_batch=16 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
   method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
   framework.replay_ratio=${RATIO} framework.log_all_vars=False \
   framework.switch_online_tasks=1  rlbench.num_vars=10

# 3 tasks -> 20x6
Query_ratio=0.3
E_LR=1e-4
C_LR=1e-4 
RATIO=3
RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-NoMatch
taskset -c 0-30 python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
replay.batch_size=20 replay.buffers_per_batch=6 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
   method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
   framework.replay_ratio=${RATIO} framework.log_all_vars=False \
   framework.switch_online_tasks=1  rlbench.num_vars=10

# 3 tasks -> 10x12
Query_ratio=0.3
E_LR=1e-4
C_LR=1e-4 
RATIO=3
RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-NoMatch
taskset -c 30-48 python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
replay.batch_size=10 replay.buffers_per_batch=12 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
   method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
   framework.replay_ratio=${RATIO} framework.log_all_vars=False \
   framework.switch_online_tasks=1  rlbench.num_vars=10




# ti5: query 3, 16 buffers
Query_size=3
E_LR=1e-4
C_LR=1e-4 
RUN=10VarEach-SmallRatio1
python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target','push_button'] rlbench.num_vars=10  \
replay.batch_size=8 replay.buffers_per_batch=16 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size}  \
   method.emb_lr=${E_LR} contexts.sampler.batch_dim=10 framework.replay_ratio=1 framework.log_all_vars=False rlbench.demo_path=/shared/mandi/all_rlbench_data

# 3 tasks 10x12, random sample action_embeddings for replay
Query_ratio=0.3
E_LR=1e-4
C_LR=1e-4 
RATIO=3
CPUS=23-48
RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-NoMatch
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
replay.batch_size=10 replay.buffers_per_batch=12 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
   method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
   framework.replay_ratio=${RATIO} framework.log_all_vars=False \
   framework.switch_online_tasks=1  rlbench.num_vars=10 contexts.agent.single_embedding_replay=False \
   rlbench.demo_path=/shared/mandi/all_rlbench_data



# rtxs1: try switching only 1 online tasks out of the 4 total:
Query_size=3
E_LR=1e-4
C_LR=1e-4 
RUN=10VarEach-SmallRatio3-Switch1
python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target','push_button'] rlbench.num_vars=10  \
replay.batch_size=8 replay.buffers_per_batch=8 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size}  \
framework.switch_online_tasks=1 rlbench.demo_path=/shared/mandi/all_rlbench_data

# 3 tasks
Query_ratio=0.3
E_LR=1e-4
C_LR=1e-4 
RATIO=3
CPUS=0-30
RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-NoMatch
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
replay.batch_size=5 replay.buffers_per_batch=24 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
   method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
   framework.replay_ratio=${RATIO} framework.log_all_vars=False \
   framework.switch_online_tasks=1  rlbench.num_vars=10  \
   rlbench.demo_path=/shared/mandi/all_rlbench_data  

# rtxl1
# 3 tasks, larger ratio
Query_ratio=0.3
E_LR=1e-4
C_LR=1e-4 
RATIO=6
CPUS=0-30
RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-NoMatch
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
replay.batch_size=10 replay.buffers_per_batch=6 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
   method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
   framework.replay_ratio=${RATIO} framework.log_all_vars=False \
   framework.switch_online_tasks=1  rlbench.num_vars=10  \
   rlbench.demo_path=/shared/mandi/all_rlbench_data  

# amd3
# 3 tasks, larger ratio, bigger emb_lr
Query_ratio=0.3
E_LR=5e-4 
RATIO=6
CPUS=30-60,90-120,180-210
RUN=10VarEach-Ratio${RATIO}-Switch1-Margin0-1-NoMatch
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup','pick_and_lift','reach_target'] rlbench.num_vars=10  \
replay.batch_size=10 replay.buffers_per_batch=12 replay.share_across_tasks=False \
 dev.encode_context=True contexts.pass_down_context=True   \
   contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.query_ratio=${Query_ratio}  \
   method.emb_lr=${E_LR} contexts.sampler.k_dim=10 \
   framework.replay_ratio=${RATIO} framework.log_all_vars=False \
   framework.switch_online_tasks=1  rlbench.num_vars=10  \
   rlbench.demo_path=/shared/mandi/all_rlbench_data  