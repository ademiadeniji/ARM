# for LR in 0.001 0.003 0.005 
# do
# for FRAME in 1 2 5
# do
# python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=sweep-frame${FRAME} \
#     dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
#     sampler.batch_dim=5 hinge_cfg.num_query=2 hinge_cfg.num_support=4 sampler.samples_per_variation=6 

# python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=sweep-frame${FRAME} \
#     dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
#     sampler.batch_dim=5 hinge_cfg.num_query=1 hinge_cfg.num_support=3 sampler.samples_per_variation=3


# done
# done

# try only one task but many variations 
for TASK in push_buttons # stack_blocks #
do
for LR in 0.001 # 0.003 
do
for FRAME in 10 1 5 2
do

RUN=1Task-${TASK}-frame${FRAME}
# python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=${RUN} \
#     dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
#     sampler.batch_dim=5 hinge_cfg.num_query=2 hinge_cfg.num_support=4 sampler.samples_per_variation=6 \
#     dataset.include_tasks=[$TASK]

python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=${RUN} \
    dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME} \
    sampler.batch_dim=5 hinge_cfg.num_query=1 hinge_cfg.num_support=3 sampler.samples_per_variation=3 \
    dataset.include_tasks=[$TASK]

done
done
done


for TASK in stack_blocks # push_buttons  
do for LR in 0.001 
do for FRAME in 10 1 5 2
do 
RUN=1Task-${TASK}-frame${FRAME} 
python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=${RUN}  dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME}   \
sampler.batch_dim=5 hinge_cfg.num_query=1 hinge_cfg.num_support=3 sampler.samples_per_variation=3  \
dataset.include_tasks=[$TASK] train.epochs=2000 train.save_freq=5000 train.vis_freq=10000 train.overwrite=True 
done
done                
done

for TASK in pick_up_cup pick_and_lift push_button lamp_on lamp_off reach_target 
do for LR in 0.001 
do for FRAME in 1 5 
do 
RUN=1Task-${TASK}-frame${FRAME} 
python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=${RUN}  dataset.num_steps_per_episode=${FRAME} encoder.DATA.NUM_FRAMES=${FRAME}   \
sampler.batch_dim=5 hinge_cfg.num_query=1 hinge_cfg.num_support=3 sampler.samples_per_variation=3  \
dataset.include_tasks=[$TASK] train.epochs=500 train.save_freq=5000 train.vis_freq=5000 train.overwrite=True 
done
done                
done