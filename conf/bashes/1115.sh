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


    RUN=10Var-Noisy-NoEncode
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_up_cup'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.encode_context=False 

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


    RUN=10Var-Noisy-Encode10
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_and_lift'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=5 

    RUN=10Var-Noisy-Encode16
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_and_lift'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=16

    RUN=10Var-Noisy-Encode20
    taskset -c $CPUS python launch_context.py run_name=${RUN} \
        tasks=['pick_and_lift'] rlbench.num_vars=10  replay.batch_size=10 dev.noisy_one_hot=True \
        dev.qnet_context_latent_size=20 