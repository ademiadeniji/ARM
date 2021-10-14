# try some RL stuff, 8vars first, context batch loads all 20 vars tho 
# ti5   
    # re-run after hacking pyrep
     FREQ=10
     Query_size=5
     E_LR=1e-4
     C_LR=1e-4 
     RUN=Handpick-8Var-BothBatch-QEncodePass
     python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
        replay.batch_size=12 replay.buffers_per_batch=5 replay.share_across_tasks=False \
        dev.handpick=[0,3,4,6,7,11,16,18] \
        dev.encode_context=True contexts.pass_down_context=True \
        run_name=${RUN} \
        rlbench.demo_path=/home/mandi/front_rlbench_data  \
        contexts.update_freq=${FREQ} \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 \
        encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=10

    FREQ=10
     Query_size=3
     E_LR=1e-4
     C_LR=1e-4 
     RUN=Handpick-8Var-BothBatch-QEncodePass
     python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
        replay.batch_size=8 replay.buffers_per_batch=8 replay.share_across_tasks=False \
        dev.handpick=[0,3,4,6,7,11,16,18] \
        dev.encode_context=True contexts.pass_down_context=True \
        run_name=${RUN} \
        rlbench.demo_path=/home/mandi/front_rlbench_data  \
        contexts.update_freq=${FREQ} \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 \
        encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=15


# rtxs1
     FREQ=100
     Query_size=5
     E_LR=1e-4
     C_LR=1e-4 
     RUN=Handpick-8Var-BothBatch-QEncodePass
     python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
        replay.batch_size=12 replay.buffers_per_batch=5 replay.share_across_tasks=False \
        dev.handpick=[0,3,4,6,7,11,16,18] \
        dev.encode_context=True contexts.pass_down_context=True \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  \
        contexts.update_freq=${FREQ} \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 \
        encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=10

    FREQ=1 
     Query_size=5
     E_LR=1e-4
     C_LR=1e-4 
     RUN=Handpick-8Var-BothBatch-QEncodePass
     python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
        replay.batch_size=12 replay.buffers_per_batch=5 replay.share_across_tasks=False \
        dev.handpick=[0,3,4,6,7,11,16,18] \
        dev.encode_context=True contexts.pass_down_context=True \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  \
        contexts.update_freq=${FREQ} \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 \
        encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=10
    
    # sim fails for 8 vars again! just do 2 vars ffs  
     FREQ=10 
     Query_size=10
     E_LR=1e-4
     C_LR=1e-4 
     RUN=2Var-BothBatch-QEncodePass
     python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=32 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/home/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ} \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20 \
    encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=20

    FREQ=10 
     Query_size=8
     E_LR=1e-4
     C_LR=1e-4 
     RUN=5Var-BothBatch-QEncodePass
     python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=18 replay.buffers_per_batch=3 replay.share_across_tasks=False \
    dev.handpick=[0,1,2,3,4] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ} \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20 \
    encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=7



# debug 
E_LR=1e-4 
Query_size=1
RUN=ReplayOnly-QEncode-Hinge${Query_size}-Emdlr${E_LR}-OneLoss
    python launch_context.py run_name=$RUN  dev.offline=True \
        dev.encode_context=True \
        contexts.update_freq=1000000 \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=False \
        replay.batch_size=6 replay.buffers_per_batch=10 \
        contexts.agent.num_query=${Query_size} \
        wandb.job_type='offline' framework.log_freq=50 \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 framework.wandb_logging=False # !!! cuz local rlbench is hacked 
 

# ti1: try pick and lift
FREQ=100000
Query_size=5
E_LR=1e-4
C_LR=1e-4 
RUN=ReplayOnly-QEncodePass
python launch_context.py tasks=['pick_and_lift']  rlbench.demos=10     \
replay.batch_size=12 replay.buffers_per_batch=10 replay.share_across_tasks=False \
dev.encode_context=True contexts.pass_down_context=True \
run_name=${RUN} \
rlbench.demo_path=/home/mandi/all_rlbench_data  \
contexts.update_freq=${FREQ} \
dev.one_hot=False \
contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
method.emb_lr=${E_LR} \
rlbench.num_vars=20 \
encoder.OPTIM.BASE_LR=${C_LR} 


# ti1 debug sim error, try not raising runtime error

# hack: vim /home/mandi/PyRep/pyrep/backend/sim.py line 339 
# pip install -e /home/mandi/PyRep/. /home/mandi/RLBench/.
FREQ=100
Query_size=5
E_LR=1e-4
C_LR=1e-4 
RUN=Handpick-8Var-BothBatch-QEncodePass
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
replay.batch_size=12 replay.buffers_per_batch=5 replay.share_across_tasks=False \
dev.handpick=[0,3,4,6,7,11,16,18] \
dev.encode_context=True contexts.pass_down_context=True \
run_name=${RUN} \
rlbench.demo_path=/shared/mandi/all_rlbench_data  \
contexts.update_freq=${FREQ} \
dev.one_hot=False \
contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
method.emb_lr=${E_LR} \
rlbench.num_vars=20 \
encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=10 framework.wandb_logging=False 


# ti1: 2var w/o context batch
    
     Query_size=10
     FREQ=100000
     E_LR=1e-4
     C_LR=1e-4 
     RUN=2Var-ReplayOnly-QEncodePass
     python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10     \
    replay.batch_size=32 replay.buffers_per_batch=2 replay.share_across_tasks=False \
    dev.handpick=[0,1] \
    dev.encode_context=True contexts.pass_down_context=True \
    run_name=${RUN} \
    rlbench.demo_path=/home/mandi/all_rlbench_data  \
    contexts.update_freq=${FREQ} \
    dev.one_hot=False \
    contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
    method.emb_lr=${E_LR} \
    rlbench.num_vars=20  contexts.sampler.samples_per_variation=10 # hack here to word around K dim 


# ti1: try 10 tasks each has 1 var
# with Context: 
Query_size=4
FREQ=10
E_LR=1e-4
C_LR=1e-4 
RUN=QEncodePass
python launch_context.py tasks='${10_tasks}' rlbench.num_vars=1 replay.batch_size=10 replay.buffers_per_batch=6 \
replay.share_across_tasks=False  dev.encode_context=True contexts.pass_down_context=True \
run_name=${RUN} rlbench.demo_path=/home/mandi/all_rlbench_data  contexts.update_freq=${FREQ} \
contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
method.emb_lr=${E_LR} encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.samples_per_variation=18

# NO Context: 
Query_size=4
FREQ=100000
E_LR=1e-4
C_LR=1e-4 
RUN=QEncodePass
python launch_context.py tasks='${10_tasks}' rlbench.num_vars=1 replay.batch_size=10 replay.buffers_per_batch=6 \
replay.share_across_tasks=False  dev.encode_context=True contexts.pass_down_context=True \
run_name=${RUN} rlbench.demo_path=/home/mandi/all_rlbench_data  contexts.update_freq=${FREQ} \
contexts.agent.replay_update=True dev.qagent_update_context=True contexts.agent.num_query=${Query_size} \
method.emb_lr=${E_LR} encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.samples_per_variation=18
