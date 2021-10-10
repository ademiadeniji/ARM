# pabti1

    # just two potentionally confusing vars : #5 and # 6 (silver and gray ) do 128 
    python launch_context.py tasks=['pick_up_cup'] run_name=Easy-2Var-Grays-OneHot-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/custom_rlbench_data \
    replay.share_across_tasks=True replay.batch_size=128 dev.handpick=[5,6]


    # hack the task again to render only one cup 
    python tools/dataset_generator.py  --tasks pick_up_cup --processes 10 --save_path /home/mandi/one_cup_data

    # 10vars
    python launch_context.py tasks=['pick_up_cup'] run_name=OneCup-10Var-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/one_cup_data \
    replay.share_across_tasks=True replay.batch_size=64 

    python launch_context.py tasks=['pick_up_cup'] run_name=OneCup-10Var-1Buffer \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True     rlbench.demo_path=/home/mandi/one_cup_data \
    replay.share_across_tasks=True replay.batch_size=128

# rtxs1:
# if prio doesn't to much for 10 might help 20 var: batch12 -> failed 
 python launch_context.py tasks=['pick_up_cup'] run_name=20Var-PRIO-10from20 \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True rlbench.demo_path=/shared/mandi/all_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=12 replay.buffers_per_batch=10 replay.update_buffer_prio=True

  python launch_context.py tasks=['pick_up_cup'] run_name=20Var-PRIO-5from20 \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True rlbench.demo_path=/shared/mandi/all_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=24 replay.buffers_per_batch=10 replay.update_buffer_prio=True



# txl1
# handpick 8 seems really good w/ prio. buffers  
RUN=Handpick-8Var-OneHot-3from8
python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
    replay.batch_size=40 rlbench.demos=5 replay.buffers_per_batch=3 replay.update_buffer_prio=True  \
    replay.share_across_tasks=False \
    dev.handpick=[0,3,4,6,7,11,16,18] \
    dev.encode_context=False contexts.pass_down_context=False \
    run_name=${RUN} \
    rlbench.demo_path=/shared/mandi/all_rlbench_data

 RUN=Handpick-8Var-OneHot-3from8
 python launch_context.py tasks=['pick_up_cup']  rlbench.demos=10 dev.one_hot=True     \
        replay.batch_size=20 rlbench.demos=5 replay.buffers_per_batch=3 replay.update_buffer_prio=True  \
        replay.share_across_tasks=False \
        dev.handpick=[0,3,4,6,7,11,16,18] \
        dev.encode_context=False contexts.pass_down_context=False \
        run_name=${RUN} \
        rlbench.demo_path=/shared/mandi/all_rlbench_data

# if prio doesn't to much for 10 might help 20 var: batch6
 python launch_context.py tasks=['pick_up_cup'] run_name=20Var-PRIO-10from20 \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    rlbench.demos=5 contexts.pass_down_context=True rlbench.demo_path=/shared/mandi/all_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=6 replay.update_buffer_prio=True 
  
  # handpick to exclude 9,10,19  -> 17Var 10from17,b12
  python launch_context.py tasks=['pick_up_cup'] run_name=17Var-PRIO-10from17 \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    contexts.pass_down_context=True rlbench.demo_path=/shared/mandi/all_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=12 replay.buffers_per_batch=10 \
    replay.update_buffer_prio=True dev.handpick=[0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18]

  # handpick to exclude 9,10,19  -> 17Var 10from17,b24
  python launch_context.py tasks=['pick_up_cup'] run_name=17Var-PRIO-10from17 \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    contexts.pass_down_context=True rlbench.demo_path=/shared/mandi/all_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=12 replay.buffers_per_batch=10 \
    replay.update_buffer_prio=True dev.handpick=[0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18]


# pabti5 
# handpick to exclude 9,10,19  -> 17Var
  python launch_context.py tasks=['pick_up_cup'] run_name=17Var-PRIO-5from17 \
    contexts.update_freq=100000 framework.training_iterations=50000 dev.one_hot=True dev.encode_context=False \
    contexts.pass_down_context=True rlbench.demo_path=/home/mandi/front_rlbench_data \
    replay.share_across_tasks=False replay.batch_size=24 replay.buffers_per_batch=5 \
    replay.update_buffer_prio=True dev.handpick=[0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18]

