# ti1: RLBench is still hacked!
# try different replay setup for Easy-10:

# debug 
  
    # 10x6  from 10 buffer
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-CatFinal-10from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True  dev.one_hot_size=10 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=6 replay.buffers_per_batch=10

    # 5x12  from 10 buffer
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-CatFinal-5from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True  dev.one_hot_size=10 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=12 replay.buffers_per_batch=5

    # 5x12  from 10 buffer WITH PRIO! 
    # buff priority 
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-PRIO-CatOnce-5from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True  dev.one_hot_size=10 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=12 replay.buffers_per_batch=5

    # 5x24  from 10 buffer WITH PRIO
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-PRIO-CatOnce-5from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True  dev.one_hot_size=10 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=24 replay.buffers_per_batch=5
    
    


    
# rtxs1: confusing 10 + 10 buffer
    python launch_context.py tasks=['pick_up_cup'] \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True \
    replay.share_across_tasks=False replay.batch_size=6 replay.buffers_per_batch=10 \
    rlbench.num_vars=10 \
    run_name=10Var-OneHot-CatOnce-10from10Buffer \
    rlbench.demo_path=/shared/mandi/all_rlbench_data framework.log_freq=200

# ti5: first 10 + PRIO -> can it get better?
    # 8x8
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=8 replay.buffers_per_batch=8  \
        dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
        rlbench.num_vars=10 run_name=PRIO-10Var-OneHot-CatFinal-8from10Buffer \
        rlbench.demo_path=/home/mandi/front_rlbench_data  dev.cat_f1=True replay.share_across_tasks=False 

    # 5x12 
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=12 replay.buffers_per_batch=5  \
        dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
        rlbench.num_vars=10  run_name=PRIO-10Var-OneHot-CatFinal-5from10Buffer \
        rlbench.demo_path=/home/mandi/front_rlbench_data  dev.cat_f1=True replay.share_across_tasks=False 

    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=21 replay.buffers_per_batch=3  \
        dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
        rlbench.num_vars=10 \
        run_name=PRIO-10Var-OneHot-CatFinal-3from10Buffer \
        rlbench.demo_path=/home/mandi/front_rlbench_data  dev.cat_f1=True replay.share_across_tasks=False 

# txl1: try some 20 vars
 # 10x12, 10x6
 python launch_context.py tasks=['pick_up_cup']  rlbench.demos=3 dev.one_hot=True     \
        replay.batch_size=12 replay.buffers_per_batch=10  \
        dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
        run_name=PRIO-20Var-OneHot-CatFinal-10from20Buffer \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  dev.cat_f1=True replay.share_across_tasks=False 
# 15x12, 15x6
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=3 dev.one_hot=True     \
        replay.batch_size=12 replay.buffers_per_batch=10  \
        dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
        run_name=PRIO-20Var-OneHot-CatFinal-15from20Buffer \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  dev.cat_f1=True replay.share_across_tasks=False 



dev.handpick=[0,3,4,6,7,10,11,16,18,19] \