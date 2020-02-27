import numpy as np 
from PIL import Image
from torch.autograd import Variable
import torch
from torchvision import transforms, datasets


totensor = transforms.ToTensor()
def shuffle_idx(data_size=10000):
    # array_idx = np.arange(data_size)
    shuffle_arr = np.arange(data_size)
    np.random.shuffle(shuffle_arr)

    return shuffle_arr


## predicated function
def load_batch_image(folder, batch_iterator, shuffle_idx, batch_size=64,):
    data_idx = shuffle_idx(dataset_size)
    num_batch = int(dataset_size / batch_size)
    print('number of batch_size', num_batch)
    dataset = []
    for batch in range(num_batch):
        batch_data = []
        for i in range(batch_size):
            image = Image.open(folder + str(batch_size*batch + i) + '.png').convert('LA')
            # print('image shape: ', image.size)
            image = np.asarray(image) 
            batch_data.append(image)
        batch_data = np.array(batch_data)
        batch_data = torch.from_numpy(batch_data)
        dataset.append(batch_data)
    return dataset
    
# load a batch of image with shuffled index.
def load_batch_image2(folder,
batch_iterator,
shuffle_idx,
batch_size=64,
start_idx=0):
    begin_idx = batch_iterator*batch_size
    batch_image = []
    # print('begin idx: ',begin_idx)
    for i in range(batch_size):
        image = Image.open(folder + str(shuffle_idx[begin_idx + i] + start_idx) + '.png').convert('LA')
        # print('image shape: ', image.size)
        image = np.asarray(image)[:,:,0] 
        batch_image.append(image)
        # dataset.append(batch_data)
        # idx+=1
    batch_image = np.array(batch_image)
    # batch_image = np.expand_dims(batch_image, axis=3)
    # batch_size = np.transpose(batch_image, (0, 3, 1, 2))
    batch_image = [totensor(i) for i in batch_image]
    # batch_image = torch.from_numpy(batch_image).permute(0,3,1,2)
    return torch.stack(batch_image, dim=0)


## use for get a batch of image from a numpy dataset
def load_batch_image3(dataset,
batch_iterator,
shuffle_idx,
batch_size=32):
    num_channels = dataset[0].shape[0]
    imsize = dataset[0].shape[1]
    begin_idx = batch_iterator*batch_size
    batch_image = np.zeros((batch_size, num_channels, imsize, imsize))
    for i in range(batch_size):
        image = dataset[i + begin_idx,:,:,:]
        # print('image shape: ', image.shape)
        batch_image[i,:,:,:] = image

    # print('shape of image: ', batch_image[0].shape)
    # batch_image = np.array(batch_image) 
    # print('shape of image: ', batch_image[0].shape)
    batch_image = [totensor(i).permute(1,0,2) for i in batch_image]
    # print('shape of image: ', batch_image[0].shape)
    return torch.stack(batch_image, dim=0)


if __name__ =='__main__':
 
    folder = '/mnt/manh/project/GAN_running/encode_image/data/kuka_grasping/kuka/'
    shuffle = shuffle_idx(10000)
    batch_image = load_batch_image2(folder=folder, batch_iterator=1, shuffle_idx=shuffle)
    x = torch.autograd.Variable(batch_image, requires_grad=False)
    print('type of batch_image ',type(batch_image))
    print(batch_image.shape)
    print(x)
 