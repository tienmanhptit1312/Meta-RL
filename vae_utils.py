import macpath
import math
import torch
import numpy as np 
import torch.nn.functional as F 

def compute_kernel(x, y):
    # print('shape of x: ', x.shape)
    # print('shape of y: ', y.shape)
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.view(x_size, 1, dim)
    y = y.view(1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)

    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def convert_to_display(samples):
    
    cnt, height, width,channels = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2], samples.shape[3]
    
    # number = sqrt(samples.shape[0])
    # illustrate = np.zeros((number, number, height, wight))
    # for i in range(number):
    #     data = sample
    #     illustrate[i,:,:,:] = sample

    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width, channels])
    # samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt, channels])
    return samples

# compute reconstruction loss
def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.shape[0]
    assert batch_size!=0, "batch size cannot equal zero"

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon,x,size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        # x_recon = F.sigmoid(x_recon)
        # x = F.sigmoid(x)
        recon_loss = F.mse_loss(x_recon, x, size_average=False) / batch_size

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.shape[0]
    assert batch_size != 0, "batch_size cannot equal zero"

    mu = mu.view(batch_size, mu.shape[1])
    logvar = logvar.view(batch_size, logvar.shape[1])

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, False)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, False)

    return total_kld, dimension_wise_kld, mean_kld