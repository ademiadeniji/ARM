# irl train:
python clip_finetune.py dataset.levels=[fail,success,expert]  dataset.sample_expert_prob=0.5 \
    run_name=LatestData-FailOnly-1x200Batch itrs=1000 dataset.nframes=1 dataset.ntraj=200


python clip_finetune.py dataset.levels=[fail,success,expert]  \
    run_name=FailOnly-1x100 itrs=1000 dataset.fail_only=True dataset.nframes=1 dataset.ntrajs=100 \
     data_path=/shared/mandi/CLIP_ARM dataset.sample_expert_prob=0.8
 
python clip_finetune.py dataset.ntrajs=20 dataset.nframes=3 lr=3e-4 \
    run_name=Iteration0-3x50Batch-FailOnly dataset.sample_expert_prob=0.5 data_path=/shared/mandi/CLIP_ARM

python clip_finetune.py dataset.levels=[fail,success,expert] \
dataset.sample_expert_prob=0.5 run_name=LatestData-FailOnly-2x150Batch itrs=1000 \
dataset.nframes=2 dataset.ntrajs=150

python clip_finetune.py task=take_lid_off_saucepan \
    prompts=["robot gripper pick and lift up saucepan lid"] \
    dataset.levels=['fail','expert'] run_name=Saucepan-Demo-Fail-3x50Batch itrs=1000 \
    dataset.nframes=3 dataset.ntrajs=50


# latest push button data: try concat both temporal and binary batch, also try predict logit architecture!
python clip_finetune.py dataset.levels=[fail,success,expert] \
dataset.sample_expert_prob=0.5 run_name=Button-Latest-ConcatBatch itrs=1000 \
    dataset.nframes=3 dataset.ntrajs=100 dataset.concat_batch=True \
    model.predict_logit=True

# R3M reward