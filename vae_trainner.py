import torch
import torch.nn as nn
import numpy as np
import multiworld
import torchvision
import gym
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from PIL import Image
from multiworld.core.image_env import ImageEnv, unormalize_image 
from rlkit.envs.vae_wrapper import VAEWrappedEnv
import cv2
import vae_nn
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import dataloader
from vae_utils import compute_mmd, convert_to_display, reconstruction_loss, kl_divergence
from os import path as osp 
# from multiworld.core.image_env import ImageEnv, unormalize_image
import os
import beta_vae

# from torch.optim import optimizer
multiworld.register_all_envs()


def get_env(env_id, init_camera, imsize=48):
    env = gym.make(env_id)
    render=False
    reward_params = dict(
        type='latent_distance'
    )
    vae = None
    image_env = ImageEnv(
        env,
        imsize,
        init_camera=init_camera,
        transpose=True,
        normalize=True,
        # grayscale=True
    )
    
    return image_env

def train_vae(
    env,
    z_dim=4,
    n_epochs=100,
    use_cuda=True,
    print_every=100,
    plot_every=500,
    batch_size=64,
    num_datasamples=200,
    encoder_param=None,
    decoder_param=None
):
    project_path = osp.abspath(os.curdir)
    parent_path = osp.abspath(osp.join(project_path, os.pardir))
    train_dataset, test_dataset = generate_vae_data(
        env,
        num_datasamples,
        0.9,
        imsize=48,
        num_channels=3)
    num_datasamples = len(train_dataset)
    ## using beta-VAE model
    # model = beta_vae.BetaVAE_B(nc=3, z_dim=4)
    ####################################
    model = vae_nn.Model(encoder_param, decoder_param)
    optimizer = torch.optim.Adam(model.parameters())
    # model = Model(z_dim, n_channels=1)
    if use_cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    i =-1
    for epoch in range(n_epochs):
        shuffle = dataloader.shuffle_idx(num_datasamples)
        num_batch = int(num_datasamples/batch_size)
        print('number of batch: ', num_batch)
        for batch_iterator in range(num_batch):
            # print('batch iterator:', batch_iterator)
            i+=1
            batch = dataloader.load_batch_image3(
                train_dataset,
                batch_iterator,
                shuffle,
                batch_size)
            optimizer.zero_grad()
            # print(batch[0].shape)
            # x = torch.autograd.Variable(batch, requires_grad=False).type(torch.float)
            x = torch.FloatTensor(batch.float()).detach()
            true_sample = torch.autograd.Variable(
                torch.randn(batch_size, z_dim),
                requires_grad=False
            )
            if use_cuda:
                x = x.cuda()
                true_sample = true_sample.cuda()
            x_reconstruction, z = model(x)
            x_reconstruction = x_reconstruction[:,:,0:48,0:48]
            # x_reconstruction, z, mu, logvar = model(x)
            # print('x_reconstruction: ', x_reconstruction[0])
            ## loss for VAE-Beta
            # recon_loss = reconstruction_loss(x=x, x_recon=x_reconstruction, distribution='gaussian')
            # kl_loss,_,_ = kl_divergence(mu, logvar)
            # loss = recon_loss + 20*kl_loss
            ########################
            mmd = compute_mmd(z, true_sample)
            nll = (x_reconstruction - x).pow(2).mean()
            loss = nll + mmd

            loss.backward()
            optimizer.step()
            # print(recon_loss)
            if i % print_every == 0:
                print("Negative log likelihood is {:.5f}, mmd loss is {:.5f}".format(
                    nll.item(), mmd.item()
                ))
            if i % plot_every == 0:
                gen_z = torch.autograd.Variable(
                    torch.randn(100, z_dim), 
                    requires_grad=False
                )
                if use_cuda:
                    gen_z = gen_z.cuda()
                samples = model.decoder(gen_z)
                print(samples.shape)
                # print(samples[0].shape)
                # samples = samples.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()
                # x_reconstructed = x_reconstructed.permute(0,2,3,1).contiguous().cpu().data.numpy()

                torchvision.utils.save_image(x_reconstruction.data,
                parent_path + '/infovae_result/reconstruct_data_' + str(i) + '.png', nrow=6, padding=6)

                torchvision.utils.save_image(samples.data,
                parent_path + '/infovae_result/sample_data_' + str(i) + '.png', nrow=10, padding=10)

                torchvision.utils.save_image(x.data,
                parent_path + '/infovae_result/input_data_' + str(i) + '.png', nrow=6, padding=6)

                # print('shape of samples: ',samples.shape)
                # # samples_result = Image.fromarray(samples)
                # # samples_result.save(project_path + 'sample_' +str(i)+'.png')
                # illustate_image = convert_to_display(samples)
                # plt.imshow(illustate_image)
                # plt.savefig(project_path + '/vae_result/samples_' + str(i) + '.png')
                # pil_image = Image.fromarray(illustate_image).convert('RGB')
                # pil_image.save(project_path + '/vae_result/samples_pil_'+ str(i) + '.png')

                # # plt.imshow(convert_to_display(x_reconstructed), cmap='Greys_r')
                # # plt.savefig(project_path + '/vae_result/reconstructed_' + str(i) + '.png')
                # # torch.save(model.state_dict(), '/mnt/manh/project/GAN_running/model/training_iteration_' + str(i) +'.pkl')
                # print('Saved a model')
            

    return model

def generate_vae_data(
    env,
    number_sample,
    test_p,
    imsize,
    num_channels,
    ):
    env.reset()
    # dataset = np.zeros((number_sample, imsize*imsize*num_channels),
    # dtype=np.float)
    dataset = np.zeros((number_sample, num_channels,imsize, imsize),
    dtype=np.uint8)
    for i in range(number_sample):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        obs = env._get_obs()
        img = obs['image_observation']
        img = img.reshape(3, imsize, imsize)
        # img = img[::-1, :, ::-1]
        # img = (img*255.0).astype(np.uint8)

        # dataset[i,:,:,:] = img
        dataset[i,:,:,:] = unormalize_image(img)
    n = int(number_sample*test_p)
    train_dataset = dataset[:n, :,:,:]
    # train_dataset = np.array(train_dataset).reshape(n,3,48,48)
    test_dataset = dataset[n:,:,:,:]
    # test_dataset = np.array(test_dataset).reshape(dataset.shape[0]-n,3,48,48)

    return train_dataset, test_dataset

if __name__=='__main__':
    env_id='SawyerPushNIPSEasy-v0'
    env = get_env(env_id, sawyer_init_camera_zoomed_in)
    project_path = osp.abspath(os.curdir)
    imsize = 48
    encoder_kwargs = dict(
        input_width=imsize,
        input_height=imsize,
        input_channels=3,
        output_size=4,
        kernel_sizes=[4,4,4],
        n_channels=[32,64,128],
        strides=[2,2,2],
        paddings=[1,1,1],
        hidden_sizes=[1024, 1024],
        batch_norm_conv=True,
        batch_norm_fc=False,
        init_w=1e-3,
        hidden_init=nn.init.xavier_uniform_,
        hidden_activation=nn.ReLU()
    )
    decoder_kwargs = dict(
        fc_input_size=4,
        hidden_sizes=[1024,1024],
        deconv_input_width=6,
        deconv_input_height=6,
        deconv_input_channels=128,
        kernel_sizes=[4,4,4],
        strides=[2,2,2],
        paddings=[1,1,1],
        n_channels=[128,64,32],
        deconv_output_channels=3,
        deconv_output_strides=1,
        deconv_output_kernel_size=2,
        batch_norm_deconv=True,
        batch_norm_fc=False,
        init_w=1e-3,
        hidden_init=nn.init.xavier_uniform_,
        hidden_activation=nn.ReLU()
    )
    # generate_vae_data(env,number_sample=100,test_p=0.9,imsize=48,num_channels=3)
    vae_model = train_vae(
        env,
        encoder_param=encoder_kwargs,
        decoder_param=decoder_kwargs
    )
    torch.save(vae_model.state_dict(), parent_path + '/vae_model/vae_model.pkl')

    # env = get_env(env_id)
    # env.reset()
    # while True:
    #     action = env.action_space.sample()
    #     obs, reward, done, _ = env.step(action)
    #     # print(obs)
    #     # image_obs = env._get_obs()
    #     print(obs)
    #     img = obs['image_observation']
    #     img = img.reshape(3, imsize, imsize).transpose()
    #     img = img[::-1, :, ::-1]
    #     img = (img*255).astype(np.uint8)
    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)


    #     # env.render()
    #     env.sim.render(100,100, camera_name='topview')