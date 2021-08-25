LR=0.003

#python train_rep.py encoder.OPTIM.BASE_LR=${LR} \
    # train.run_name=sweepSize sampler.batch_dim=6 sampler.samples_per_variation=5 hinge_cfg.num_support=4 hinge_cfg.num_query=1;


#python train_rep.py encoder.OPTIM.BASE_LR=${LR} \
    # train.run_name=sweepSize sampler.batch_dim=6 sampler.samples_per_variation=5 hinge_cfg.num_support=3 hinge_cfg.num_query=2 ;

#python train_rep.py encoder.OPTIM.BASE_LR=${LR} \
    # train.run_name=sweepSize sampler.batch_dim=3 sampler.samples_per_variation=10 hinge_cfg.num_support=8 hinge_cfg.num_query=2;

# python train_rep.py encoder.OPTIM.BASE_LR=${LR} \
#     train.run_name=sweepSize sampler.batch_dim=6 \
#     sampler.samples_per_variation=10 hinge_cfg.num_support=6 hinge_cfg.num_query=4 

for FRAME in 10
do
python train_rep.py encoder.OPTIM.BASE_LR=${LR} \
    train.run_name=sweepModel-frame${FRAME} \
    dataset.num_steps_per_episode=${FRAME} \
    encoder.DATA.NUM_FRAMES=${FRAME} sampler.batch_dim=5 

python train_rep.py encoder.OPTIM.BASE_LR=${LR} \
    train.run_name=sweepModel-frame${FRAME} \
    dataset.num_steps_per_episode=${FRAME} \
    encoder.DATA.NUM_FRAMES=${FRAME} sampler.batch_dim=10 


done