Task=[take_lid_off_saucepan]
DIM=16
HID=2048 
RATIO=6 
for DEMO in 0 1 5 10
do 
    RUN=Scratch-1Cam-dVAE-3frameStack-Ratio${RATIO}
    taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=${Task} \
    framework.replay_ratio=$RATIO tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True \
    contexts.loss_mode='dvae' dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
    rlbench.demos=${DEMO} framework.ckpt_eval=True replay.batch_size=60 replay.buffers_per_batch=1 \
    tasks.train_steps=5000
done


# pick up cup has more vars!
DIM=16
HID=2048 
RATIO=6 
for DEMO in 0 1 5 10
do 
    RUN=Scratch-1Cam-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
    taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=${Task} \
    framework.replay_ratio=$RATIO tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True \
    contexts.loss_mode='dvae' dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
    rlbench.demos=${DEMO} framework.ckpt_eval=True tasks.train_steps=5000 
done
