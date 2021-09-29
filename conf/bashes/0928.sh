# ti1: messed with the task code!
# now has only 10 var!
python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True \
    rlbench.demo_path=/home/mandi/custom_rlbench_data dev.one_hot_size=10

python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True replay.batch_size=128 \
    rlbench.demo_path=/home/mandi/custom_rlbench_data dev.one_hot_size=10

python launch_context.py tasks=['pick_up_cup'] run_name=Easy-10Var-OneHot-1Buffer-2e4lr \
    contexts.update_freq=100000 framework.training_iterations=50000 \
    dev.one_hot=True  dev.one_hot_size=20 dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True replay.batch_size=128 \
    rlbench.demo_path=/home/mandi/custom_rlbench_data dev.one_hot_size=10 method.lr=2e-4 


# rtxs1 Handpicked 10 distinct tasks
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
 replay.batch_size=128 \
 dev.one_hot_size=20 dev.encode_context=False contexts.pass_down_context=False \
 rlbench.num_vars=10 dev.handpick=[0,3,4,6,7,10,11,16,18,19] \
 run_name=10Var-Handpick-OneHot-CatOnce-1Buffer \
 rlbench.demo_path=/shared/mandi/all_rlbench_data  