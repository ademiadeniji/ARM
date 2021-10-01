# sudo fuser -k /dev/nvidia3 
# python tools/dataset_generator.py  --tasks pick_up_cup --processes 10 --save_path /home/mandi/custom_rlbench_data

#ti1: messed with the task code!
# now has only 10 var! -> RERUN b64
python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True \
    rlbench.demo_path=/home/mandi/custom_rlbench_data dev.one_hot_size=10

python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-CatFinal-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True \
    rlbench.demo_path=/home/mandi/custom_rlbench_data dev.one_hot_size=10 dev.cat_f1=True

python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-CatOnce-lr7e4-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True \
    rlbench.demo_path=/home/mandi/custom_rlbench_data dev.one_hot_size=10 method.lr=7e-4


# rtxs1 
   # Handpicked 10 distinct tasks
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    replay.batch_size=128 \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
    run_name=10Var-Handpick-OneHot-CatOnce-1Buffer \
    rlbench.demo_path=/shared/mandi/all_rlbench_data  
   # no handpick, batch 64 
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=10 \
    run_name=10Var-OneHot-CatOnce-1Buffer \
    rlbench.demo_path=/shared/mandi/all_rlbench_data


# ti5 Handpicked 10 distinct tasks, cat final
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    replay.batch_size=128 \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
    run_name=10Var-Handpick-OneHot-CatFinal-1Buffer \
    rlbench.demo_path=/home/mandi/front_rlbench_data  dev.cat_f1=True

    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    replay.batch_size=128 \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
    run_name=10Var-Handpick-OneHot-CatOnce-lr2e4-1Buffer \
    rlbench.demo_path=/home/mandi/front_rlbench_data method.lr=2e-4
    # small batch 
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    replay.batch_size=64 \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
    run_name=10Var-Handpick-OneHot-CatOnce-1Buffer \
    rlbench.demo_path=/home/mandi/front_rlbench_data 
    # hand pick only 5! 
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    dev.handpick=[0,3,4,6,10] run_name=5Var-Handpick-OneHot-CatOnce-1Buffer \
    rlbench.demo_path=/home/mandi/front_rlbench_data framework.log_freq=200 

    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    dev.handpick=[0,3,4,6,10] run_name=5Var-Handpick-OneHot-CatFinal-1Buffer \
    rlbench.demo_path=/home/mandi/front_rlbench_data framework.log_freq=200 dev.cat_f1=True 

    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
    dev.handpick=[0,3,4,6,10] run_name=5Var-Handpick-OneHot-CatOnce-1Buffer \
    rlbench.demo_path=/home/mandi/front_rlbench_data framework.log_freq=200 replay.batch_size=32 



# txl1: Handpicked 10 distinct tasks, bsize 256 Conv3D -> not as good, overfit?
# NOTE: changed  the conv3d feature size from 64 to 32!
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
 replay.batch_size=256 dev.conv3d=True \
 dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
 rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
 run_name=10Var-Handpick-Conv3d-OneHot-CatOnce-1Buffer \
 rlbench.demo_path=/shared/mandi/all_rlbench_data  

python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
 replay.batch_size=256 dev.conv3d=True \
 dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
 rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
 run_name=10Var-Handpick-Conv3d-lr1e4-OneHot-CatOnce-1Buffer \
 rlbench.demo_path=/shared/mandi/all_rlbench_data method.lr=1e-4

python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
 replay.batch_size=256 dev.conv3d=True \
 dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
 run_name=20Var-Conv3d-f32-lr2e4-OneHot-CatOnce-1Buffer \
 rlbench.demo_path=/shared/mandi/all_rlbench_data method.lr=2e-4


# txl1: 
    # Handpicked 10 distinct tasks, bsize 64 cat final 
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=64 \
        dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
        rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        run_name=10Var-Handpick-OneHot-CatFinal-1Buffer \
        rlbench.demo_path=/shared/mandi/all_rlbench_data dev.cat_f1=True
    # cut down LATENT_SIZE in launch_utils to 32 for Conv3d
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=256 \
        dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
        rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        run_name=10Var-Handpick-Conv3d-f32-OneHot-CatOnce-1Buffer \
        rlbench.demo_path=/shared/mandi/all_rlbench_data 