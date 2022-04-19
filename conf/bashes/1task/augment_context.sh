Task=pick_up_cup
DIM=16
HID=2048 
RATIO=30
DEMO=1
ASIZE=6
RUN=AugBatch${ASIZE}-1Cam-dVAE-3frameStack-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=$RATIO tasks.demo_length=3 dataset.defer_transforms=True dev.discrete=True \
contexts.loss_mode='dvae' dev.qnet_context_latent_size=${DIM} dev.encode_context_hidden=${HID} \
rlbench.demos=${DEMO} replay.batch_size=6 \
replay.buffers_per_batch=10 rlbench.num_vars=-1 \
tasks.train_steps=100000 replay.replay_size=100000 dev.augment_batch=2 \
dev.augment_batch=${ASIZE} 

Task=pick_and_lift
RATIO=30
DEMO=3
ASIZE=6
RUN=AugReward-AugBatch${ASIZE}-OneHot-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=$RATIO rlbench.demos=${DEMO}  replay.batch_size=6 \
replay.buffers_per_batch=10 rlbench.num_vars=-1 dev.one_hot=True \
tasks.train_steps=100000 replay.replay_size=100000 dev.augment_batch=${ASIZE} dev.augment_reward=True 

 
Task=pick_up_cup
DEMO=1
Task=pick_and_lift
RATIO=30
DEMO=3
RUN=AugReward-NoAugBatch-OneHot-Ratio${RATIO}-Demo${DEMO}
taskset -c $CPUS python launch_context.py run_name=$RUN tasks=Single_1cam tasks.all_tasks=[${Task}] \
framework.replay_ratio=$RATIO rlbench.demos=${DEMO}  replay.batch_size=6 \
replay.buffers_per_batch=20 rlbench.num_vars=-1 dev.one_hot=True \
tasks.train_steps=100000 replay.replay_size=100000 dev.augment_reward=True 