# pabti1: noisy one-hot 
    RUN=10Var-Noisy-Encode5
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=5 

    RUN=10Var-Noisy-Encode10
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=10 

    RUN=10Var-Noisy-Encode20
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=20 


    RUN=10Var-Noisy-NoEncode # -> worst performing
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.encode_context=False 
    
    RUN=10Var-Hingev2-Emb8-QEncode16
    Query_ratio=0.3
    K_dim=10 
    MAR=5e-1
    E_LR=5e-4
    WEI=1
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=16  \
    contexts.agent.query_ratio=${Query_ratio} method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.training_iterations=20000 contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    contexts.agent.loss_mode='hinge-v2' 

# pabti5: noisy one-hot 1 layer 
    RUN=10Var-Noisy-Encode5-1Layer
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=5 dev.single_layer_context=True \
        rlbench.demo_path=/shared/mandi/all_rlbench_data 

    RUN=10Var-Noisy-NoEncode-1Layer
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.encode_context=False dev.single_layer_context=True \
        rlbench.demo_path=/shared/mandi/all_rlbench_data 

# pabamd2: hinge v2 


    RUN=10Var-Hingev2-Emb8-QEncode16
    Query_ratio=0.3
    K_dim=10 
    MAR=5e-2
    E_LR=5e-4
    WEI=1
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=16  \
    contexts.agent.query_ratio=${Query_ratio} method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.training_iterations=20000 contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    contexts.agent.loss_mode='hinge-v2'  rlbench.demo_path=/shared/mandi/all_rlbench_data 

    RUN=10Var-Hingev2-Emb8-QEncode16-1Layer
    Query_ratio=0.3
    K_dim=10
    MAR=5e-2
    E_LR=5e-4
    WEI=1
    RUN=10Var-Hingev2-Emb8-QEncode16-1Layer-MAR${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=16  \
    contexts.agent.query_ratio=${Query_ratio} method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.training_iterations=20000 contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    contexts.agent.loss_mode='hinge-v2'  rlbench.demo_path=/shared/mandi/all_rlbench_data dev.single_layer_context=True

    
    Query_ratio=0.3
    K_dim=10
    MAR=1e-2
    E_LR=5e-4
    WEI=1
    RUN=10Var-Hingev2-Emb8-QEncode16-1Layer-MAR${MAR}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10 \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=16  \
    contexts.agent.query_ratio=${Query_ratio} method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.training_iterations=20000 contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    contexts.agent.loss_mode='hinge-v2'  rlbench.demo_path=/shared/mandi/all_rlbench_data dev.single_layer_context=True


    # expand enc. dim.:
    Query_ratio=0.3
    K_dim=10 
    RATIO=6 
    MAR=5e-1
    E_LR=5e-4
    WEI=1
    RUN=10Var-Emd8-Encode20-Normed-Mean-Margin${MAR}-Elr${E_LR}-Weight${WEI}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 \
    dev.qagent_update_context=True  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False \
    dev.encode_context=False contexts.emb_weight=${WEI}   \
    rlbench.demo_path=/shared/mandi/all_rlbench_data 
 

# pabtxl1: noisy one-hot ratio 6
    RUN=10Var-Noisy-Encode5-Ratio6
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=5 framework.replay_ratio=6 rlbench.demo_path=/shared/mandi/all_rlbench_data 

    # expand enc. dim.:
    Query_ratio=0.3
    K_dim=10 
    RATIO=6 
    MAR=5e-1
    E_LR=5e-4
    WEI=1
    RUN=10Var-Emd8-Encode16-Normed-1Layer-Single-Margin${MAR}-Elr${E_LR}-Weight${WEI}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=16  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False contexts.emb_weight=${WEI}   \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  dev.single_layer_context=True

# pabti1  expand enc. layer to 2! 
    Query_ratio=0.3
    K_dim=10 
    RATIO=6 
    MAR=5e-1
    E_LR=5e-4
    WEI=1
    RUN=10Var-Emd16-2MLPEncode16-Normed-1Layer-Single-Margin${MAR}-Elr${E_LR}-Weight${WEI}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 dev.encode_context=True dev.qnet_context_latent_size=16  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False contexts.emb_weight=${WEI}   \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    dev.single_layer_context=True dev.qnet_2_layer_context=True 

    Query_ratio=0.3
    K_dim=10 
    RATIO=6 
    MAR=5e-1
    E_LR=5e-4
    WEI=1
    RUN=10Var-Emd16-2MLPEncode8-Normed-1Layer-Single-Margin${MAR}-Elr${E_LR}-Weight${WEI}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=4 \
    dev.encode_context=True dev.qnet_context_latent_size=8  \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False contexts.emb_weight=${WEI}   \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    dev.single_layer_context=True dev.qnet_2_layer_context=True 

    Query_ratio=0.3
    K_dim=10 
    RATIO=6 
    MAR=5e-1
    E_LR=5e-4
    WEI=1
    RUN=10Var-Emd8-2MLPEncode16-Normed-1Layer-Single-Margin${MAR}-Elr${E_LR}-Weight${WEI}
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
    tasks=['pick_up_cup'] rlbench.num_vars=10  \
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 \
    dev.qnet_context_latent_size=16 \
    contexts.agent.query_ratio=${Query_ratio} \
    method.emb_lr=${E_LR} contexts.sampler.k_dim=${K_dim} \
    framework.replay_ratio=${RATIO} framework.training_iterations=20000 \
    contexts.agent.margin=${MAR} replay.update_buffer_prio=False contexts.emb_weight=${WEI}   \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  \
    dev.single_layer_context=True dev.qnet_2_layer_context=True 

    # Act also uses norm!! 
    RUN=10Var-Emd8-Encode16-SupportMean-1Layer
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  \
        encoder.MODEL.OUT_DIM=2 dev.qnet_context_latent_size=16 \
        framework.training_iterations=30000 \
        replay.update_buffer_prio=False \
        dev.single_layer_context=True 

    RUN=10Var-Emd16-Encode8-SupportMean-1Layer
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  \
        encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
        framework.training_iterations=30000 \
        replay.update_buffer_prio=False \
        dev.single_layer_context=True 

    RUN=10Var-Emd8-2MLPEncode16-SupportMean-1Layer-
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  \
        encoder.MODEL.OUT_DIM=2 dev.qnet_context_latent_size=16 \
        framework.training_iterations=30000 \
        replay.update_buffer_prio=False \
        dev.single_layer_context=True dev.qnet_2_layer_context=True 
    
    RUN=10Var-Emd8-2MLPEncode8-1Layer-Single
    taskset -c $CPUS python launch_context.py run_name=${RUN}  \
    tasks=['pick_up_cup'] rlbench.num_vars=10  \
    encoder.MODEL.OUT_DIM=8 dev.qnet_context_latent_size=8 dev.single_layer_context=True \
     replay.update_buffer_prio=False  \
     dev.qnet_2_layer_context=True  framework.training_iterations=50000


    RUN=10Var-Resume-Emd8-2MLPEncode16-Normed-1Layer-Single
    taskset -c $CPUS python launch_context.py run_name=${RUN}  \
    tasks=['pick_up_cup'] rlbench.num_vars=10  \
    encoder.MODEL.OUT_DIM=8 dev.qnet_context_latent_size=8 dev.single_layer_context=True \
     replay.update_buffer_prio=False  \
     dev.qnet_2_layer_context=True  framework.training_iterations=20000 \
     resume=True resume_run=pick_up_cup-20var/10Var-Emd8-2MLPEncode16-Normed-1Layer-Single-Margin5e-1-Elr5e-4-Weight1-Replay_B10x6-Q0.3/seed0 \
    resume_step=3400

# pabti5: margin 0.5 + SupportMean
    RUN=10Var-Emd8-2MLPEncode16-SupportMean-1Layer-Freeze
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  \
        encoder.MODEL.OUT_DIM=2 dev.qnet_context_latent_size=16 \
        framework.training_iterations=20000 \
        replay.update_buffer_prio=False \
        dev.single_layer_context=True dev.qnet_2_layer_context=True \
        contexts.pretrain_replay_steps=2000 dev.freeze_emb=True 
        
        
    RUN=10Var-Emd8-2MLPEncode16-SupportMean-1Layer-NoHinge
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  \
        encoder.MODEL.OUT_DIM=2 dev.qnet_context_latent_size=16 \
        framework.training_iterations=20000 \
        replay.update_buffer_prio=False \
        dev.single_layer_context=True \
        contexts.emb_weight=0

# txl1 
    RUN=10Var-Emd16-Encode8-SupportMean-1Layer-Freeze
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  \
        encoder.MODEL.OUT_DIM=4 dev.qnet_context_latent_size=8 \
        framework.training_iterations=20000 \
        replay.update_buffer_prio=False \
        dev.single_layer_context=True dev.qnet_2_layer_context=True \
        contexts.pretrain_replay_steps=2000 dev.freeze_emb=True 
        


