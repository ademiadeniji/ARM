#   fix the error priority to avg across entire buffer 

# pabti1
# fix the error priority 
    # 5x24  from 10 buffer WITH PRIO-Avg
    
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-PRIO-Avg-CatOnce-5from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=24 replay.buffers_per_batch=5

    # try 1 buffer at a time 
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-PRIO-Avg-CatOnce-1from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True  rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=64 replay.buffers_per_batch=1

# ti5: 
    # handpick 10 + AvgPRIO -> 5x12 
    RUN=Handpick-10Var-OneHot-PRIO-Avg-CatOnce-5from10Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=12 replay.buffers_per_batch=5 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        rlbench.num_vars=10  run_name=${RUN} \
        rlbench.demo_path=/home/mandi/front_rlbench_data  

    # handpick 10 + AvgPRIO -> 1x64 
    RUN=Handpick-10Var-OneHot-PRIO-Avg-CatOnce-1from10Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=12 replay.buffers_per_batch=1 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        rlbench.num_vars=10  run_name=${RUN} \
        rlbench.demo_path=/home/mandi/front_rlbench_data  

# rtxs1
    # handpick 10 + AvgPRIO -> 5x12 
    RUN=Handpick-10Var-OneHot-PRIO-Avg-CatOnce-5from10Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=24 replay.buffers_per_batch=5 \
        replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        rlbench.num_vars=10  run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  
