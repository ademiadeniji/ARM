# try barely updating context with hinge loss

# ti1:
for FREQ in 100 500 
do 
for ITR in 1  
do
python launch_context.py tasks=['pick_up_cup'] run_name=HingeUpdate-lr1e-4-batch20x3 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 encoder.OPTIM.BASE_LR=1e-4 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR} framework.training_iterations=30000

done
done
# no hingle loss at all
FREQ=50000
ITR=1 
python launch_context.py tasks=['pick_up_cup'] run_name=NoHinge \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR} framework.training_iterations=50000
# small emb
FREQ=100
ITR=1 
python launch_context.py tasks=['pick_up_cup'] run_name=SmallHinge-lr3e-4-batch20x3 \
encoder.MODEL.OUT_DIM=16 contexts.agent.embedding_size=64 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 encoder.OPTIM.BASE_LR=3e-4 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR} framework.training_iterations=50000

FREQ=100
ITR=1 
python launch_context.py tasks=['pick_up_cup'] run_name=SmallHinge-lr3e-4-batch20x3 \
encoder.MODEL.OUT_DIM=8 contexts.agent.embedding_size=32 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 encoder.OPTIM.BASE_LR=3e-4 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR} framework.training_iterations=50000




for FREQ in 100  
do 
for ITR in 1  
do
python launch_context.py tasks=['pick_up_cup','pick_and_lift'] run_name=HingeUpdate-lr1e-4-batch20x3 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 encoder.OPTIM.BASE_LR=1e-4 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR} framework.training_iterations=50000 \
framework.log_freq=500
done
done

# rtxs1 
for FREQ in 1000 
do 
for ITR in 1  
do
python launch_context.py tasks=['pick_up_cup','pick_and_lift'] run_name=HingeUpdate-lr1e-4-batch20x3 \
contexts.sampler.batch_dim=20 contexts.sampler.samples_per_variation=3 encoder.OPTIM.BASE_LR=1e-4 \
contexts.update_freq=${FREQ} contexts.num_update_itrs=${ITR} framework.training_iterations=50000 \
framework.log_freq=500 rlbench.demo_path=/shared/mandi/all_rlbench_data
done
done

# ti5: update context a lot at beginning
 
# ti5: pretrain-contex t
FREQ=100
ITR=10000

python launch_context.py tasks=['pick_up_cup'] run_name=PretrainHithHinge-lr1e-3-batch10x5 \
contexts.sampler.batch_dim=10 contexts.sampler.samples_per_variation=5 encoder.OPTIM.BASE_LR=1e-3 \
contexts.agent.num_support=4 contexts.agent.num_query=1 \
contexts.update_freq=${FREQ} contexts.pretrain_context_steps=${ITR} \
framework.log_freq=100 rlbench.demo_path=/home/mandi/front_rlbench_data ;

python launch_context.py tasks=['pick_up_cup'] run_name=PretrainHinge-lr1e-3-batch5x5 \
contexts.sampler.batch_dim=5 contexts.sampler.samples_per_variation=5 encoder.OPTIM.BASE_LR=1e-3 \
contexts.agent.num_support=4 contexts.agent.num_query=1 \
contexts.update_freq=${FREQ} contexts.pretrain_context_steps=${ITR} \
framework.log_freq=100 rlbench.demo_path=/home/mandi/front_rlbench_data ;