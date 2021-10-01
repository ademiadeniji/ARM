# ti1: RLBench is still hacked!
# try different replay setup for Easy-10:
    # 10x6  from 10 buffer
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-CatFinal-10from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True  dev.one_hot_size=10 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=6 dev.buffers_per_batch=10

    # 5x12  from 10 buffer
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-CatFinal-10from10Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True  dev.one_hot_size=10 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=12 dev.buffers_per_batch=5

    # batchsize 128 from 1 buffer
    # buff priority 
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-burn \
        contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True  dev.one_hot_size=10 dev.encode_context=False \
        framework.wandb_logging=False rlbench.demos=1 rlbench.demo_path=/home/mandi/custom_rlbench_data replay.share_across_tasks=False 

    
# rtxs1: confusing 10 + 10 buffer
    python launch_context.py tasks=['pick_up_cup'] \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True \
    replay.share_across_tasks=False replay.batch_size=6 dev.buffers_per_batch=10 \
    rlbench.num_vars=10 \
    run_name=10Var-OneHot-CatOnce-10from10Buffer \
    rlbench.demo_path=/shared/mandi/all_rlbench_data framework.log_freq=200

# ti5: b128 for 5var 