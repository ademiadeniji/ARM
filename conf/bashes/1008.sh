# just no env runner, no context dataset, just add hinge loss whenever doing replay batch 
# since no EnvRunner, just use original dataset  


# 1. replay batch only, fix batch size 60, either  
#     A) both TD error and hinge update embed net; or B) detach context embedding when calculating Q-update, 
#     vary 1) different lr for embnet; 2) different batch size 3) whether to q-encode the given embedding 
for E_LR in 1e-3 1e-4 5e-4 
do 
for Query_size in 1 3
do 
    RUN=ReplayOnly-NoQEncode-Hinge${Query_size}-Emdlr${E_LR}-BothLoss
    python launch_context.py run_name=$RUN  dev.offline=True \
        contexts.update_freq=1000000 \
        dev.one_hot=False dev.encode_context=False \
        contexts.agent.replay_update=True \
        replay.batch_size=6 replay.buffers_per_batch=10 \
        contexts.agent.num_query=${Query_size} \
        dev.qagent_update_context=True \
        wandb.job_type='offline' framework.log_freq=50 \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 # !!! cuz local rlbench is hacked 

done
done 

# only update hinge loss from replay batch 
for E_LR in 1e-3 1e-4 5e-4 
do 
for Query_size in 1 3
do 
    RUN=ReplayOnly-NoQEncode-Hinge${Query_size}-Emdlr${E_LR}-OneLoss
    python launch_context.py run_name=$RUN  dev.offline=True \
        contexts.update_freq=1000000 \
        dev.one_hot=False dev.encode_context=False \
        contexts.agent.replay_update=True dev.qagent_update_context=False \
        replay.batch_size=6 replay.buffers_per_batch=10 \
        contexts.agent.num_query=${Query_size} \
        wandb.job_type='offline' framework.log_freq=50 \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 # !!! cuz local rlbench is hacked 

done
done 


# encode again in Qnet, both loss 
for E_LR in 1e-3 1e-4 5e-4 
do 
for Query_size in 1 3
do 
    RUN=ReplayOnly-QEncode-Hinge${Query_size}-Emdlr${E_LR}-BothLoss
    python launch_context.py run_name=$RUN  dev.offline=True \
        dev.encode_context=True \
        contexts.update_freq=1000000 \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=True \
        replay.batch_size=6 replay.buffers_per_batch=10 \
        contexts.agent.num_query=${Query_size} \
        wandb.job_type='offline' framework.log_freq=50 \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 # !!! cuz local rlbench is hacked 

done
done 


# encode again in Qnet, one loss 
for E_LR in 1e-3 1e-4 5e-4 
do 
for Query_size in 1 3
do 
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
        rlbench.num_vars=20 # !!! cuz local rlbench is hacked 

done
done 

# 10/10: add context batch, fix replay batch to 64, QEncode=True, and num_query=3, 
# vary encoder.OPTIM.lr, context batch size,  

# also do 10from20 buffers from Replay only 
Query_size=5
for E_LR in 1e-4 3e-4 
do 
    RUN=ReplayOnly-QEncode-Hinge${Query_size}-Emdlr${E_LR}-OneLoss
    python launch_context.py run_name=$RUN  dev.offline=True \
        dev.encode_context=True \
        contexts.update_freq=1000000 \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=False \
        replay.batch_size=12 replay.buffers_per_batch=10 \
        contexts.agent.num_query=${Query_size} \
        wandb.job_type='offline' framework.log_freq=100 \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 # !!! cuz local rlbench is hacked 

done
done 

Query_size=5
for E_LR in 1e-4 3e-4 
do 
    RUN=ReplayOnly-NoQEncode-Hinge${Query_size}-Emdlr${E_LR}-OneLoss
    python launch_context.py run_name=$RUN  dev.offline=True \
        dev.encode_context=False \
        contexts.update_freq=1000000 \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=False \
        replay.batch_size=12 replay.buffers_per_batch=10 \
        contexts.agent.num_query=${Query_size} \
        wandb.job_type='offline' framework.log_freq=100 \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20  # !!! cuz local rlbench is hacked 

done
done 



E_LR=1e-4 
Query_size=3
for C_LR in 1e-4 3e-4 1e-3 
do 
for FREQ in 1 10 100
do
 RUN=BothBatch-BothLoss-QEncode-Hinge${Query_size}-Emdlr${E_LR}-Conlr${C_LR}-B20
 python launch_context.py run_name=$RUN  dev.offline=True \
        dev.encode_context=True \
        contexts.update_freq=${FREQ} \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=False \
        replay.batch_size=6 replay.buffers_per_batch=10 \
        contexts.agent.num_query=${Query_size} \
        wandb.job_type='offline' framework.log_freq=50 \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 \
        encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=20 
done 
done 
 
E_LR=1e-4 
Query_size=3
for C_LR in 1e-4 3e-4 1e-3 
do 
for FREQ in 1 10 100
do
 RUN=BothBatch-BothLoss-QEncode-Hinge${Query_size}-Emdlr${E_LR}-Conlr${C_LR}-B10
 python launch_context.py run_name=$RUN  dev.offline=True \
        dev.encode_context=True \
        contexts.update_freq=${FREQ} \
        dev.one_hot=False \
        contexts.agent.replay_update=True dev.qagent_update_context=False \
        replay.batch_size=6 replay.buffers_per_batch=10 \
        contexts.agent.num_query=${Query_size} \
        wandb.job_type='offline' framework.log_freq=50 \
        method.emb_lr=${E_LR} \
        rlbench.num_vars=20 \
        encoder.OPTIM.BASE_LR=${C_LR} contexts.sampler.batch_dim=10 
done 
done 