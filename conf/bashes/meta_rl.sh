# sweep lr and freq
for FREQ in 1 10 100
do 
for ITR in 1 10 100 
do
python launch_context.py tasks=['pick_up_cup'] run_name=hinge-lr1e-3-batch10x3 \
contexts.sampler.batch_dim=10 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR};

python launch_context.py tasks=['pick_up_cup'] run_name=hinge-lr1e-3-batch20x3 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR}

done
done

for FREQ in 1 10 100
do 
for ITR in 1 10 100 
do
python launch_context.py tasks=['pick_up_cup','pick_and_lift'] run_name=hinge-lr1e-3-batch10x3 \
contexts.sampler.batch_dim=10 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR};

python launch_context.py tasks=['pick_up_cup','pick_and_lift'] run_name=hinge-lr1e-3-batch20x3 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR}

done
done


# pure MT:

python launch_context.py tasks=['pick_up_cup'] run_name=MultiTask mt_only=True 