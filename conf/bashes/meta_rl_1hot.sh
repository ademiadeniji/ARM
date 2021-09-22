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

# small batcb
# ti5
python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-NoQEncode-NoPassDown-lr1e3 \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  \
replay.batch_size=32 method.lr=1e-3 rlbench.demo_path=/home/mandi/front_rlbench_data 

# ti1:
python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-NoQEncode-NoPassDown-lr7e-4 \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  \
replay.batch_size=32 method.lr=7e-4

python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-NoQEncode-NoPassDown-lr7e-4 \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  \
replay.batch_size=48 method.lr=7e-4

# dev: cat more 
python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-CatALL-NoQEncode-NoPassDown \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  replay.batch_size=96

python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-CatALL-NoQEncode-NoPassDown \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  replay.batch_size=64

# ti5: cat more
python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-CatALL-lr1e3-NoQEncode-NoPassDown \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  replay.batch_size=64 method.lr=1e-3
# rtxs1: cat more
python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-CatALL-lr7e4-NoQEncode-NoPassDown \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  replay.batch_size=64 method.lr=7e-4  rlbench.demo_path=/shared/mandi/all_rlbench_data
# txl1
python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-CatNoFinal-NoQEncode-NoPassDown \
contexts.update_freq=100000 framework.training_iterations=50000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_f1=False  


# dev: 
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False replay.batch_size=3 run_name=OneHot-CatALL-20Buffer \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False 

python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False run_name=OneHot-CatALL-10Buffer \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False dev.buffers_per_batch=10 replay.batch_size=6