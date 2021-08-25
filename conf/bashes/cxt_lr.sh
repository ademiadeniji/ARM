for LR in 0.001 0.003 0.005 0.1
do
for FRAME in 5 1
do
python train_rep.py encoder.OPTIM.BASE_LR=${LR} train.run_name=sweepLR-frame${FRAME} \
dataset.num_steps_per_episode=${FRAME} \
encoder.DATA.NUM_FRAMES=${FRAME} sampler.batch_dim=10

done
done
