rtxs1:
# try freeze + encode to smaller/bigger emb size -> neither works better than no-pretrain 
Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=10Var-Emd32-QEncode16-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10  \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=8 contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data dev.qnet_context_latent_size=16

Query_ratio=0.3
E_LR=5e-4
K_dim=16 
RATIO=6
MAR=5e-2 
WEI=1
RUN=10Var-Emd16-QEncode8-Joint-EmbWeight_${WEI}-Pretrain5k-Freeze
taskset -c $CPUS python launch_context.py run_name=${RUN} \
tasks=['pick_up_cup'] rlbench.num_vars=10  \
dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 contexts.pretrain_replay_steps=5000 dev.freeze_emb=True \
contexts.agent.query_ratio=${Query_ratio}  \
method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
rlbench.demo_path=/shared/mandi/all_rlbench_data dev.qnet_context_latent_size=8

# txl1 
# try cat_final and cat_up1
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=12
    MAR=5e-2 
    WEI=1
    RUN=10Var-Emd32-QEncode16-CatUp1-Joint-EmbWeight_${WEI}-Pretrain2k-Freeze
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=12 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=8 contexts.pretrain_replay_steps=2000 dev.freeze_emb=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.qnet_context_latent_size=16 dev.cat_up1=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=12
    MAR=5e-2 
    WEI=1
    RUN=10Var-Emd32-QEncode16-CatFinal-Joint-EmbWeight_${WEI}-Pretrain2k-Freeze
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=12 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=8 contexts.pretrain_replay_steps=2000 dev.freeze_emb=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.qnet_context_latent_size=16 dev.cat_final=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=12
    MAR=5e-2 
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatUp1-Joint-EmbWeight_${WEI}-Pretrain2k-Freeze
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=12 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.pretrain_replay_steps=2000 dev.freeze_emb=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=12
    MAR=5e-2 
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatFinal-Joint-EmbWeight_${WEI}-Pretrain2k-Freeze
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=12 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.pretrain_replay_steps=2000 dev.freeze_emb=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.qnet_context_latent_size=16 dev.cat_final=True 

# ti5: no pretrain
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=12
    MAR=5e-2 
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatUp1-Joint-EmbWeight_${WEI} 
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=12
    MAR=5e-2 
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatF1-Joint-EmbWeight_${WEI} 
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_f1=True 

# rtxs1: no pre-train but bigger margin
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=1e-1
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatUp1-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatUp1-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 

# amd2: no pre-train but bigger margin
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=1
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatUp1-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 

    Query_ratio=0.3
    E_LR=3e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatUp1-Margin${MAR}-EmbLR${E_LR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 

    Query_ratio=0.3
    E_LR=1e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatUp1-Margin${MAR}-EmbLR${E_LR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 



# txl1: no Q encode, big margin 
    Query_ratio=0.3
    E_LR=3e-4
    K_dim=16 
    RATIO=12
    MAR=1 
    WEI=1
    RUN=10Var-Emd16-NoQEncode-CatUp1-Joint-EmbWeight_${WEI}-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=12 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 

    Query_ratio=0.3
    E_LR=3e-4
    K_dim=16 
    RATIO=12
    MAR=1 
    WEI=1
    RUN=10Var-Emd8-NoQEncode-CatUp1-Joint-EmbWeight_${WEI}-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=12 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True 


# ti1: no second layer context  
    RUN=10Var-OneHot-1Layer
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 framework.training_iterations=30000 \
    replay.update_buffer_prio=False \
    dev.one_hot=True dev.single_layer_context=True 

    RUN=10Var-OneHot-1Layer
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_and_lift'] rlbench.num_vars=10 framework.training_iterations=30000 \
    replay.update_buffer_prio=False \
    dev.one_hot=True dev.single_layer_context=True 

    # pretrain + small size
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=1
    WEI=1
    RUN=10Var-Emd8-NoQEncode-EmbWeight_${WEI}-Margin${MAR}-Freeze-1Layer
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 contexts.pretrain_replay_steps=3000 dev.freeze_emb=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}  dev.single_layer_context=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=1
    WEI=1
    RUN=10Var-Emd4-NoQEncode-EmbWeight_${WEI}-CatUp1-Margin${MAR}-Freeze
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=8 contexts.pretrain_replay_steps=3000 dev.freeze_emb=True \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=30000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.pass_down_context=True contexts.emb_weight=${WEI}  dev.cat_up1=True 

    # try the single layer thing on joint 
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-2 
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatF1-1Layer  
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 \
    dev.qnet_context_latent_size=8 \
    contexts.agent.query_ratio=${Query_ratio}  \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=True contexts.pass_down_context=True contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_f1=True \
    dev.single_layer_context=True 

# txl1: cat twicee + QEncode
    Query_ratio=0.3
    E_LR=3e-4
    K_dim=16 
    RATIO=12
    MAR=1
    WEI=1
    RUN=10Var-Emd16-QEncode8-CatTwice-Joint-EmbWeight_${WEI}-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=12 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_up1=True dev.cat_f1=True 



# amd2 FiLM:
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd64-FiLM-Joint-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=6 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=16  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.use_film=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd32-FiLM-Joint-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=6 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.use_film=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd8-FiLM-Joint-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.use_film=True 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd8-FiLM-Joint-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.use_film=True 

# rtxs1 FiLM + pre-train
    Query_ratio=0.3
    E_LR=1e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd32-FiLM-Joint-Margin${MAR}-Pretrain2k
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=6 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.use_film=True 

    # FiLM + one-hot
     
    RATIO=6
    RUN=10Var-OneHot-FiLM-Joint-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} tasks=['pick_and_lift'] \
    rlbench.num_vars=10  replay.batch_size=6  dev.one_hot=True framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.use_film=True 

# txl1 FiLM + freeze:
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd32-FiLM-Joint-Margin${MAR}-Pretrain2k
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=6 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.use_film=True dev.freeze_emb=True


    RATIO=6
    RUN=10Var-OneHot-FiLM-Joint-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} tasks=['pick_up_cup'] \
    rlbench.num_vars=10  replay.batch_size=10  dev.one_hot=True framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    rlbench.demo_path=/shared/mandi/all_rlbench_data dev.use_film=True contexts.pretrain_replay_steps=2000

    
# ti1: normalized:
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd16-Encode8-Normed-Single-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    contexts.emb_weight=${WEI} 

    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd16-Encode8-Normed-Mean-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} contexts.agent.single_embedding_replay=False 

     
    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd8-Encode16-Normed-Mean-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=16  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} contexts.agent.single_embedding_replay=False 


    Query_ratio=0.3
    E_LR=5e-4
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    RUN=10Var-Emd8-NoEncode-Normed-Mean-Margin${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=False \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} contexts.agent.single_embedding_replay=False 

# ti5: try diff lr, margin
    Query_ratio=0.3
    K_dim=16 
    RATIO=6
    MAR=5e-1
    WEI=1
    for E_LR in 3e-4 1e-4
    do
    RUN=10Var-Emd16-Encode8-Normed-Mean-Margin${MAR}-Elr${E_LR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} contexts.agent.single_embedding_replay=False  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data 

    done

    Query_ratio=0.3
    K_dim=16 
    RATIO=6 
    WEI=1
    E_LR=3e-4
    for MAR in 5e-2 1
    do
    RUN=10Var-Emd16-Encode8-Normed-Mean-Margin${MAR}-Elr${E_LR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} contexts.agent.single_embedding_replay=False  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data 
    done

# rtxs1
    Query_ratio=0.3
    K_dim=16 
    RATIO=6 
    MAR=5e-1
    E_LR=3e-4
    for WEI in 10 0.1
    do
    RUN=10Var-Emd16-Encode8-Normed-Mean-Margin${MAR}-Elr${E_LR}-Weight${WEI}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} contexts.agent.single_embedding_replay=False  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data 
    done

    Query_ratio=0.3
    K_dim=16 
    RATIO=6 
    MAR=5e-1
    E_LR=3e-4
    WEI=1
    RUN=10Var-Emd16-Encode8-Normed-Single-Margin${MAR}-Elr${E_LR}-Weight${WEI}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=20 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} contexts.agent.single_embedding_replay=True \
    rlbench.demo_path=/shared/mandi/all_rlbench_data 


Query_ratio=0.3
    K_dim=16 
    RATIO=6 
    MAR=5e-1
    E_LR=5e-4
    for WEI in 100 
    do
    RUN=10Var-Emd16-Encode8-Normed-Mean-Margin${MAR}-Elr${E_LR}-Weight${WEI}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=10000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI} contexts.agent.single_embedding_replay=False  \
    rlbench.demo_path=/shared/mandi/all_rlbench_data 
    done