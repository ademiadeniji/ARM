# pabti5 
python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-QEncode \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=True \
rlbench.demos=0 \
contexts.pass_down_context=True framework.wandb_logging=False rlbench.demo_path=/home/mandi/front_rlbench_data


python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-NoQEncode \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False \
rlbench.demos=1 contexts.pass_down_context=True framework.wandb_logging=False \
rlbench.demo_path=/home/mandi/front_rlbench_data