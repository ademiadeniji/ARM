#   fix the error priority to avg across entire buffer 

# pabti1
# fix the error priority 
    # 5x24  from 10 buffer WITH PRIO-Avg
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-PRIO-Avg-CatOnce-5from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=24 replay.buffers_per_batch=5

    # 5x12  from 10 buffer WITH PRIO-Avg
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-PRIO-Avg-CatOnce-5from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=12 replay.buffers_per_batch=5

    # just two potentionally confusing vars : #5 and # 6 (silver and gray )
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-2Var-Grays-OneHot-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=True replay.batch_size=64 dev.handpick=[5,6]

    # try 1 buffer at a time 
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-PRIO-Avg-CatOnce-1from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True  rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=64 replay.buffers_per_batch=1

    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-10from10 \
     contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=10 contexts.pass_down_context=True  rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=6 replay.buffers_per_batch=10
 

    # 8Var, exclude the gray ones 
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-8Var-1Buffer \
     contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=10 contexts.pass_down_context=True  rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=True replay.batch_size=64 dev.handpick=[0,1,2,3,4,7,8,9]

    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-8Var-1Buffer \
     contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=10 contexts.pass_down_context=True  rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=True replay.batch_size=128 dev.handpick=[0,1,2,3,4,7,8,9]



# ti5: 
    # handpick 10 + AvgPRIO -> 5x12 
    RUN=Handpick-10Var-OneHot-PRIO-Avg-CatOnce-5from10Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=12 replay.buffers_per_batch=5 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/home/mandi/front_rlbench_data  

    # handpick 10 + AvgPRIO -> 1x64 
    RUN=Handpick-10Var-OneHot-PRIO-Avg-CatOnce-1from10Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=12 replay.buffers_per_batch=1 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/home/mandi/front_rlbench_data  

# rtxs1
    # handpick 10 + AvgPRIO -> 5x24 
    RUN=Handpick-10Var-OneHot-PRIO-Avg-CatOnce-5from10Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=24 replay.buffers_per_batch=5 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  
    # handpick 8:
    RUN=Handpick-8Var-OneHot-1Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=64 replay.share_across_tasks=True  \
        dev.handpick=[0,3,4,6,7,11,16,18] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data
    

# txl1: batch64 worked for no-priotization cases anyway
    # handpick 8: b128
    RUN=Handpick-8Var-OneHot-1Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=128 replay.share_across_tasks=True  \
        dev.handpick=[0,3,4,6,7,11,16,18] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data

    # handpick 3 x 21
    RUN=Handpick-10Var-OneHot-UNIF-CatOnce-3from10
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=21 replay.buffers_per_batch=3 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data replay.update_buffer_prio=False 

    RUN=Handpick-10Var-OneHot-PRIO-Avg-CatOnce-3from10
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=21 replay.buffers_per_batch=3 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data replay.update_buffer_prio=True 

    # handpick uniform 5 buffers x 24 
    RUN=Handpick-10Var-OneHot-UNIF-CatOnce-5from10Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=24 replay.buffers_per_batch=5 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data replay.update_buffer_prio=False 
    
    # handpick 10 + 1 AvgPRIO -> 1x120 
    RUN=Handpick-10Var-OneHot-PRIO-Avg-CatOnce-1from10Buffer
    python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=120 replay.buffers_per_batch=1 replay.share_across_tasks=False  \
        dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data  
