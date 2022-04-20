# rl train:
python launch.py rew.scale_logits=True run_name=TunedItr0-Aug3 \
    rew.model='Iteration0-5x20Batch' rew.step=450 rew.save_data=True rew.use_aug=True rew.aug_avg=3 

MODEL=Iteration0-3x40Batch-SmoothSuccess
STEP=400

MODEL=Iteration0-3x50Batch-FailOnly
STEP=900
RUN=TunedItr0-FailOnly
python launch.py rew.scale_logits=True run_name=$RUN \
    rew.model=$MODEL rew.step=$STEP rew.save_data=False rew.use_aug=True


# irl train:
python clip_finetune.py dataset.levels=[fail,success,expert]  dataset.sample_expert_prob=0.5 \
    run_name=DemoAndFail itrs=1000

python clip_finetune.py dataset.levels=[fail,success,expert]  \
    run_name=FailOnly-1x100 itrs=1000 dataset.fail_only=True dataset.nframes=1 dataset.ntrajs=100 \
     data_path=/shared/mandi/CLIP_ARM dataset.sample_expert_prob=0.8
 
python clip_finetune.py dataset.ntrajs=20 dataset.nframes=3 lr=3e-4 \
    run_name=Iteration0-3x50Batch-FailOnly dataset.sample_expert_prob=0.5 data_path=/shared/mandi/CLIP_ARM

