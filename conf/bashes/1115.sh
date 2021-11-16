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

# pabamd2: noisy one-hot but size 20 


    RUN=10Var-Noisy20-Encode10
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_and_lift'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.noisy_dim_20=True \
        dev.qnet_context_latent_size=5 rlbench.demo_path=/shared/mandi/all_rlbench_data 

    RUN=10Var-Noisy20-Encode16
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_and_lift'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.noisy_dim_20=True \
        dev.qnet_context_latent_size=16 rlbench.demo_path=/shared/mandi/all_rlbench_data 

    RUN=10Var-Noisy20-Encode20
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_and_lift'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.noisy_dim_20=True \
        dev.qnet_context_latent_size=20 rlbench.demo_path=/shared/mandi/all_rlbench_data 
    
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
    dev.qagent_update_context=True encoder.MODEL.OUT_DIM=2 dev.encode_context=True dev.qnet_context_latent_size=20  \
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

    RUN=10Var-Noisy20-Encode16-Ratio6
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=5 framework.replay_ratio=6 \
        dev.noisy_dim_20=True \
        dev.qnet_context_latent_size=16 rlbench.demo_path=/shared/mandi/all_rlbench_data 

