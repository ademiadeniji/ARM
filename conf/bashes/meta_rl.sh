# sweep lr and freq
for FREQ in 1 10 100
do 
for ITR in 1 10  
do
python launch_context.py tasks=['pick_up_cup'] run_name=Update-hinge-lr1e-3-batch10x3 \
contexts.sampler.batch_dim=10 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR};

done
done

# python launch_context.py tasks=['pick_up_cup'] run_name=Update-hinge-lr1e-3-batch20x3 \
# contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 \
# contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR}


# ti5, no updating context 
ITR=10
for FREQ in 10 100
do
python launch_context.py tasks=['pick_up_cup'] run_name=NoQUpdateContext-hinge-lr1e-4-batch20x3 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 \
encoder.OPTIM.BASE_LR=1e-4 contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR} \
dev.qagent_update_context=False rlbench.demo_path=/home/mandi/front_rlbench_data
done


# rtxs1, no updating context 
FREQ=1
ITR=10
python launch_context.py tasks=['pick_up_cup'] run_name=NoQUpdateContext-hinge-lr3e-4-batch10x3 \
contexts.sampler.batch_dim=10 contexts.sampler.samples_per_variation=3 \
encoder.OPTIM.BASE_LR=3e-4 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR} dev.qagent_update_context=False rlbench.demo_path=/shared/mandi/all_rlbench_data


for FREQ in 1 10 100
do 
for ITR in 1 10  
do
python launch_context.py tasks=['pick_up_cup','pick_and_lift'] run_name=Update-hinge-lr1e-3-batch10x3 \
contexts.sampler.batch_dim=10 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR};

python launch_context.py tasks=['pick_up_cup','pick_and_lift'] run_name=Update-hinge-lr1e-3-batch20x3 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR}

done
done


# pure MT:

python launch_context.py tasks=['pick_up_cup'] run_name=MultiTask mt_only=True; python launch_context.py tasks=['pick_up_cup','pick_and_lift'] run_name=MultiTask mt_only=True

# pabti5: push buttons
python launch_context.py tasks=['push_buttons'] run_name=MultiTask mt_only=True;

for FREQ in 10  
do 
for ITR in 1 
do
python launch_context.py tasks=['push_buttons'] run_name=Update-hinge-lr1e-3-batch10x3 \
contexts.sampler.batch_dim=10 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR};

python launch_context.py tasks=['push_buttons'] run_name=Update-hinge-lr1e-3-batch20x3 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR}

done
done

# scratch baselines: pabti5
for TASK in take_lid_off_saucepan reach_target press_switch
do
for DEMO in 1 0 10
do  
 for BEFORE in 100 200
do
    python launch_context.py tasks=[$TASK]  run_name=Baseline-NoContext-lr5e4 mt_only=True \
    framework.training_iterations=5000 framework.transitions_before_train=${BEFORE} \
    rlbench.demo_path=/home/mandi/front_rlbench_data
    rlbench.demos=${DEMO} replay.batch_size=64 framework.replay_ratio=64 method.lr=5e-4 

    done
done
done