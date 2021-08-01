7_task_wine:    [pick_up_cup,phone_on_base,pick_and_lift,put_rubbish_in_bin,reach_target,take_lid_off_saucepan,take_umbrella_out_of_umbrella_stand]
7_task_target: [pick_up_cup,phone_on_base,pick_and_lift,put_rubbish_in_bin,stack_wine,take_lid_off_saucepan,take_umbrella_out_of_umbrella_stand]
7_task_phone: [pick_up_cup,stack_wine,pick_and_lift,put_rubbish_in_bin,reach_target,take_lid_off_saucepan,take_umbrella_out_of_umbrella_stand]
tasks: ???
short_names: ??? 
run_name: burn 
log_path: ??? 
rlbench:
    task: take_lid_off_saucepan
    tasks: ${tasks}
    demos: 10
    demo_path: /home/mandi/_ARM_MultiTask_Example/data
    episode_length: 10
    cameras: [front]
    camera_resolution: [128, 128]
    scene_bounds: [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]

replay:
    batch_size: 9
    timesteps: 1
    prioritisation: True
    use_disk: False
    path: '/tmp/arm_sanity/replay'  # Only used when use_disk is True.

framework:
    log_freq: 100
    train_envs: 2
    eval_envs: 2
    replay_ratio: 128
    transitions_before_train: 100
    tensorboard_logging: True
    csv_logging: False
    training_iterations: 100000
    gpu: 0
    logdir: '/home/mandi/ARM/log/'
    seeds: 1

defaults:
    - method: C2FARM

hydra:
    run:
        dir: ${framework.logdir}/ #${rlbench.task}/${method.name}
