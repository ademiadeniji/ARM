# rl train:
python launch.py rew.scale_logits=True

# irl train:
python clip_finetune.py dataset.levels=[fail,expert]  dataset.sample_expert_prob=0.5 \
    run_name=DemoAndFail itrs=500