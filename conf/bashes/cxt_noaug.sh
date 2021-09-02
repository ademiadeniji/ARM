for LR in 0.001 0.003 0.005 
do
for FRAME in 1 5 
do
python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=NoCrop-frame${FRAME} \
    dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
    sampler.batch_dim=5 hinge_cfg.num_query=2 hinge_cfg.num_support=4 sampler.samples_per_variation=6 \
    dataset.data_augs.weak_crop_scale=[1,1] dataset.data_augs.weak_crop_ratio=[1,1] train.overwrite=True 

python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=HeavyCrop-frame${FRAME} \
    dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
    sampler.batch_dim=5 hinge_cfg.num_query=2  hinge_cfg.num_support=4 sampler.samples_per_variation=6 \
    dataset.data_augs.weak_crop_scale=[0.5,0.8] dataset.data_augs.weak_crop_ratio=[0.7,0.9] train.overwrite=True 

done
done
