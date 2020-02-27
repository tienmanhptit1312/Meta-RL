import numpy as np 
import torch.nn as nn
import torchvision
import cv2
from os import path as osp
import os
from PIL import Image
import timeit
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.util.io import load_local_or_remote_file
from rlkit.util.video import dump_video
import gym
import multiworld
from rlkit.torch.vae.vae_trainer import ConvVAETrainer

from multiworld.core.image_env import ImageEnv, unormalize_image
from rlkit.samplers.data_collector.vae_env import VAEWrappedEnvPathCollector
from rlkit.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm

multiworld.register_all_envs()


def skewfit_full_experiment(variant):
    # variant['skewfit_variant']['save_vae_data'] =  True
    # full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    skewfit_experiment(variant)

def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    skewfit_variant = variant['skewfit_variant']
    env_id = variant['env_id']
    init_camera = variant.get('init_camera', None)
    image_size = variant.get('image_size', 84)

# def train_vae_and_update_variant(variant):
#     from rlkit.core import logger

def train_vae(variant, return_data=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule
    from rlkit.torch.vae.conv_vae import ConvVAE as conv_vae
    from rlkit.torch.vae.conv_vae import ConvVAE
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu 
    from rlkit.pythonplusplus import identity
    import torch
    from rlkit.torch.vae.conv_vae import imsize48_default_architecture

    beta = variant["beta"]
    representation_size = variant.get("representation_size", 4)
    
    train_dataset, test_dataset = generate_vae_data(variant)
    decoder_activation = identity
    # train_dataset = train_dataset.cuda()
    # test_dataset = test_dataset.cuda()
    architecture = variant.get('vae_architecture', imsize48_default_architecture)
    image_size = variant.get('image_size', 48)
    input_channels = variant.get('input_channels', 1)
    vae_model = ConvVAE(
        representation_size,
        decoder_output_activation = decoder_activation,
        architecture = architecture,
        imsize = image_size,
        input_channels=input_channels,
        decoder_distribution='gaussian_identity_variance'
    )

    vae_model.cuda()
    vae_trainner = ConvVAETrainer(
        train_dataset,
        test_dataset,
        vae_model,
        beta=beta,
        beta_schedule=None
    )
    save_period = variant['save_period']
    dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)
    for epoch in range(variant['num_epochs']):
        vae_trainner.train_epoch(epoch)
        vae_trainner.test_epoch(epoch)
        if epoch % save_period == 0:
            vae_trainner.dump_samples(epoch)
        vae_trainner.update_train_weights()
    # logger.save_extra_data(vae_model, 'vae.pkl', mode='pickle')
    project_path = osp.abspath(os.curdir)
    save_dir = osp.join(project_path + str('/saved_model/'), 'vae_model.pkl')
    torch.save(vae_model.state_dict(), save_dir)
    # torch.save(vae_model.state_dict(), \
    # '/mnt/manh/project/visual_RL_imaged_goal/saved_model/vae_model.pkl')
    if return_data:
        return vae_model, train_dataset, test_dataset
    return vae_model

def generate_vae_data(variant):
    env_id = variant.get('env_id', None)
    N = variant.get('N', 1000)
    test_p = variant.get('test_p', 0.9)
    image_size = variant.get('image_size', 84)
    num_channels = variant.get('num_channels', 3)
    init_camera = variant.get('init_camera', None)
    oracle_dataset_using_set_to_goal = variant.get(
        'oracle_dataset_using_set_to_goal', False)
    random_rollout_data = variant.get('random_rollout_data', False)
    random_and_oracle_policy_data = variant.get(
        'random_and_oracle_policy_data', False)
    random_and_oracle_policy_data_split = variant.get(
        'random_and_oracle_policy_data_split', 0)
    n_random_steps = variant.get('n_random_steps', 100)
    show = variant.get('show', False)

    
    import rlkit.torch.pytorch_util as ptu 

    info = {}

    
    env = gym.make(env_id)
    env = ImageEnv(
        env,
        image_size,
        init_camera=init_camera,
        transpose=True,
        normalize=True, 
        non_presampled_goal_img_is_garbage=None
    )

    dataset = np.zeros((N, image_size*image_size*num_channels),
    dtype=np.uint8)
    # print('aa')
    for i in range(N):
        if oracle_dataset_using_set_to_goal:
            goal = env.sample_goal()
            print(goal)
            env.set_to_goal(goal)
            obs = env._get_obs()
            # print(obs)
            img = obs['image_observation']
            print('length of image arr:', len(img))
            ### this block to test image #############################
            if show:
                # print(obs['image_observation'])
                img = img.reshape(3, image_size, image_size).transpose()
                print(img.size)
                img = img[::-1, :, ::-1]
                img = (img*255).astype(np.uint8)
                img = Image.fromarray(img, 'RGB')
                print(img.size)
                # print(len(img))
                # img.save('/home/manhlt/extra_disk/data/RIG_data/image_'+str(i)+'.png')
            #############################################################
            dataset[i:] = unormalize_image(img)
    n = int(N * test_p)
    train_dataset = dataset[:n, : ]
    test_dataset = dataset[n:, : ]
    return train_dataset, test_dataset 

def train_vae_and_update_variant(variant):
    from rlkit.core import logger
    vae_model, train_dataset, test_dataset = train_vae(variant, return_data=True)
    variant['vae_train_data'] = train_dataset
    variant['vae_test_data'] = test_dataset
    ## save vae model
    # logger.save_extra_data(vae_model, 'vae_model.pkl', mode='pickle')
    # logger.remove_tabular_output(
    #     'vae_progress.csv',
    #     relative_to_snapshot_dir=True,
    # )
    variant['vae_model'] = vae_model

def get_envs(variant):
    from rlkit.envs.vae_wrapper import VAEWrappedEnv
    env = gym.make(variant['env_id'])
    render = variant.get('render', False)
    reward_params = variant.get("reward_params", dict(
        type='latent_distance'
    ))
    init_camera = variant['init_camera']
    image_env = ImageEnv(
        env,
        variant.get('image_size', 48),
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    vae = variant['vae_model']
    vae_env = VAEWrappedEnv(
        image_env,
        vae,
        imsize=variant.get('image_size', 48),
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        reward_params=reward_params,
        sample_from_true_prior=True
    )
    env = vae_env
    return env

def skewfit_experiment(variant):
    import rlkit.torch.pytorch_util as ptu 
    from rlkit.data_management.online_vae_replay_buffer \
        import OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    import rlkit.torch.vae.vae_schedules as vae_schedules

    #### getting parameter for training VAE and RIG
    env = get_envs(variant)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
        + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    replay_buffer_kwargs = variant.get('replay_buffer_kwargs', dict(
        start_skew_epoch=10,
        max_size=int(100000),
        fraction_goals_rollout_goals=0.2,
        fraction_goals_env_goals=0.5,
        exploration_rewards_type='None',
        vae_priority_type='vae_prob',
        priority_function_kwargs=dict(
            sampling_method='importance_sampling',
            decoder_distribution='gaussian_identity_variance',
            num_latents_to_sample=10,),
        power=0,
        relabeling_goal_sampling_mode='vae_prior',
    ))
    online_vae_trainer_kwargs = variant.get('online_vae_trainer_kwargs', dict(
        beta=20,
        lr=1e-3
    ))
    max_path_length = variant.get('max_path_length', 50)
    algo_kwargs = variant.get('algo_kwargs', dict(
        batch_size=1024,
        num_epochs=1000,
        num_eval_steps_per_epoch=500,
        num_expl_steps_per_train_loop=500,
        num_trains_per_train_loop=1000,
        min_num_steps_before_training=10000,
        vae_training_schedule=vae_schedules.custom_schedule_2,
        oracle_data=False,
        vae_save_period=50,
        parallel_vae_train=False,
    ))
    twin_sac_trainer_kwargs = variant.get('twin_sac_trainer_kwargs', dict(
        discount=0.99,
        reward_scale=1,
        soft_target_tau=1e-3,
        target_update_period=1,  # 1
        use_automatic_entropy_tuning=True,
    ))
    ############################################################################

    qf1 = FlattenMlp(
        input_size = obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes
    )
    qf2 = FlattenMlp(
        input_size = obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes
    )
    target_qf2 = FlattenMlp(
        input_size= obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes
    )

    vae = variant['vae_model']
    # create a replay buffer for training an online VAE
    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **replay_buffer_kwargs
    )
    # create an online vae_trainer to train vae on the fly
    vae_trainer = ConvVAETrainer(
        variant['vae_train_data'],
        variant['vae_test_data'],
        vae,
        **online_vae_trainer_kwargs
    )
    # create a SACTrainer to learn a soft Q-function and appropriate policy
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **twin_sac_trainer_kwargs
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        variant.get('evaluation_goal_sampling_mode', 'reset_of_env'),
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        variant.get('exploration_goal_sampling_mode', 'vae_prior'),
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        max_path_length=max_path_length,
        **algo_kwargs
    )

    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()




#################################################################################
if __name__ =='__main__':
    from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
    variant = dict(
        env_id = 'SawyerPushNIPSEasy-v0',
        init_camera = sawyer_init_camera_zoomed_in,
        oracle_dataset_using_set_to_goal = True,
        image_size=48,
        beta=20,
        save_period=25,
        num_epochs=10,
        input_channels=3,
        custom_goal_sampler='replay_buffer'
    )
    mode = 'local'
    # train_vae_and_update_variant(variant)
    from rlkit.launchers.launcher_util import run_experiment
    run_experiment(
        skewfit_full_experiment,
        mode=mode,
        variant=variant,
        use_gpu=True,
        num_exps_per_instance=3,
        gcp_kwargs=dict(
            terminate=True,
            zone='us-east1-c',
            gpu_kwargs=dict(
            gpu_model='nvidia-tesla-k80',
            num_gpu=1)
                )
    )
    # skewfit_full_experiment(variant)


