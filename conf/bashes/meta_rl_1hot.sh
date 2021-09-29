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

# one buff cat twice, big batch
python launch_context.py tasks=['pick_up_cup'] run_name=OneHot-CatTwice-NoQEncode-NoPassDown \
contexts.update_freq=100000 framework.training_iterations=30000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  replay.batch_size=128

python launch_context.py tasks=['pick_and_lift'] run_name=OneHot-CatTwice-20Bufer-NoQEncode-NoPassDown \
contexts.update_freq=100000 framework.training_iterations=30000 \
dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
rlbench.demos=1  replay.batch_size=6 replay.share_across_tasks=False

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

# dev: 3dconv block instead of inception
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=64 dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=2 dev.conv3d=True 
# txl1
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=256 dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    dev.conv3d=True run_name=Conv3d-OneHot-1Buffer 
# ti1: batch 196 smaller lr 
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=196 dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    dev.conv3d=True run_name=Conv3d-OneHot-1Buffer-lr1e4 method.lr=1e-4
# ti1: batch 128 cat twice + conv3d
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=128 dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    dev.conv3d=True run_name=Conv3d-CatTwice-OneHot-1Buffer dev.cat_up1=True


# dev: single var + cat
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=64 run_name=2Var-OneHot-CatOnce-1Buffer \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False rlbench.num_vars=2
# ti5
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=64 run_name=2Var-OneHot-CatTwice-1Buffer \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=2 rlbench.demo_path=/home/mandi/front_rlbench_data dev.cat_up1=True 

python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=64 run_name=2Var-OneHot-CatTwice-2Buffer \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=2 rlbench.demo_path=/home/mandi/front_rlbench_data dev.cat_up1=True \
    replay.share_across_tasks=False dev.buffers_per_batch=2 

# rtxs1
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=128 run_name=2Var-OneHot-CatOnce-1Buffer \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=2 rlbench.demo_path=/shared/mandi/all_rlbench_data 
# txl1
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True  \
    replay.batch_size=128 run_name=10Var-OneHot-CatOnce-1Buffer \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=10 rlbench.demo_path=/shared/mandi/all_rlbench_data 


# dev: Multi-Buffer, ti1
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False replay.batch_size=3 run_name=OneHot-CatALL-20Buffer \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False 

python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False run_name=OneHot-CatALL-10Buffer \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False dev.buffers_per_batch=10 replay.batch_size=6
# dev: Multi-Buffer, 
# rtxs1 1e-3 -> bad 
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False replay.batch_size=3 run_name=OneHot-CatALL-20Buffer-1e3 \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False method.lr=1e-3 \
    rlbench.demo_path=/shared/mandi/all_rlbench_data ; 
# bigger replay size
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False replay.batch_size=3 run_name=OneHot-CatALL-20Buffer-BIG \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.demo_path=/shared/mandi/all_rlbench_data replay.replay_size=2e6

# txl1 , big bsize
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False replay.batch_size=5 run_name=OneHot-CatALL-20Buffer \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.demo_path=/shared/mandi/all_rlbench_data; \
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True \
    replay.share_across_tasks=False run_name=OneHot-CatALL-10Buffer \
        dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
        rlbench.demo_path=/shared/mandi/all_rlbench_data dev.buffers_per_batch=10 replay.batch_size=10  

python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False replay.batch_size=5 run_name=OneHot-CatALL-20Buffer-3e4- \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.demo_path=/shared/mandi/all_rlbench_data method.lr=3e-4 

# ti5: 5 buffers only
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False run_name=OneHot-CatALL-5Buffer \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.demo_path=/home/mandi/front_rlbench_data dev.buffers_per_batch=5 replay.batch_size=12
# bigger replay size
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False replay.batch_size=5 run_name=OneHot-CatALL-20Buffer-BIG- \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.demo_path=/home/mandi/front_rlbench_data replay.replay_size=2e6 framework.training_iterations=30000

python launch_context.py tasks=['pick_up_cup']  rlbench.demos=1 dev.one_hot=True  \
    replay.share_across_tasks=False replay.batch_size=5 run_name=OneHot-CatALL-20Buffer-BIG-lr1e4 \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    method.lr=1e-4 \
    rlbench.demo_path=/home/mandi/front_rlbench_data replay.replay_size=2e6 framework.training_iterations=30000
