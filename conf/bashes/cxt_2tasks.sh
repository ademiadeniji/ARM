# try only one task but many variations 
TASKS=light_bulb_in,light_bulb_out
for LR in 0.001 #0.003 
do
for FRAME in 2 5 10 1 
do

RUN=2Task-light_bulb_in-light_bulb_out-frame${FRAME}

# python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=${RUN} \
#     dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
#     sampler.batch_dim=5 hinge_cfg.num_query=2 hinge_cfg.num_support=4 sampler.samples_per_variation=6 \
#     dataset.include_tasks=[${TASKS}]

python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=${RUN} \
    dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
    sampler.batch_dim=5 hinge_cfg.num_query=1 hinge_cfg.num_support=3 sampler.samples_per_variation=3 \
    dataset.include_tasks=[${TASKS}] \
    train.epochs=2000 train.save_freq=5000 train.vis_freq=10000 train.overwrite=True 
done
done


TASKS=[block_pyramid,stack_blocks]
for LR in 0.001 #0.003 
do
for FRAME in 5 10 1 2
do

RUN=2Task-block_pyramid-stack_blocks-frame${FRAME}

# python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=${RUN} \
#     dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
#     sampler.batch_dim=5 hinge_cfg.num_query=2 hinge_cfg.num_support=4 sampler.samples_per_variation=6 \
#     dataset.include_tasks=${TASKS}

python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=${RUN} \
    dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
    sampler.batch_dim=5 hinge_cfg.num_query=1 hinge_cfg.num_support=3 sampler.samples_per_variation=3 \
    dataset.include_tasks=${TASKS} \
    train.epochs=2000 train.save_freq=5000 train.vis_freq=10000 train.overwrite=True 

done
done