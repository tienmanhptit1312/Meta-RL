import torch 
from torch import nn as nn
import numpy as np 




class CNNEncoder(nn.Module):
    def __init__(self,
    input_width,
    input_height,
    input_channels,
    output_size,
    kernel_sizes,
    n_channels,
    strides,
    paddings,
    hidden_sizes=None,
    added_fc_input_size=0,
    batch_norm_conv=False,
    batch_norm_fc=False,
    init_w=1e-4,
    hidden_init=nn.init.xavier_uniform_,
    hidden_activation=nn.ReLU()):
        super(CNNEncoder, self).__init__()
        if hidden_sizes is None:
            self.hidden_sizes = []
        else:
            self.hidden_sizes = hidden_sizes
        assert (len(kernel_sizes) == \
            len(n_channels) == \
            len(strides) == \
            len(paddings)), "size of kernel, n_channels, strides, and paddings is not equal"

        
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.strides = strides
        self.paddings = paddings
        self.added_fc_input_size = added_fc_input_size
        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_fc = batch_norm_fc
        self.hidden_int = hidden_init
        self.hidden_activation = hidden_activation

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for output_channel, kernel_size, stride, padding in zip(self.n_channels, self.kernel_sizes, self.strides, self.paddings):

            conv = nn.Conv2d(self.input_channels,
                            output_channel,
                            kernel_size,
                            stride=stride,
                            padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            self.conv_layers.append(conv)
            self.input_channels = output_channel

        # a test matrix to compute number channels of CNN 
        test_mat = torch.zeros(1, input_channels, self.input_width, self.input_height)
        
        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)
            self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            
        fc_input_size = int(np.prod(test_mat.shape))

        for idx, hidden_size in enumerate(self.hidden_sizes):

            fc_layer = nn.Linear(fc_input_size, hidden_size)
            fc_norm_layer = nn.BatchNorm1d(hidden_size)

            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            self.fc_norm_layers.append(fc_norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, self.output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self,x):
        h = self.apply_forward(x, self.conv_layers, self.conv_norm_layers,
        use_batch_norm=self.batch_norm_conv)

        h = h.view(h.size(0), -1)

        h= self.apply_forward(h, self.fc_layers, self.fc_norm_layers,
        use_batch_norm=self.batch_norm_fc)

        output = self.last_fc(h)
        return output

    def apply_forward(self,
                    input,
                    hidden_layers,
                    norm_layers,
                    use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h

class DCNNDecoder(nn.Module):
    def __init__(self,
    fc_input_size,
    hidden_sizes,

    deconv_input_width,
    deconv_input_height,
    deconv_input_channels,

    deconv_output_kernel_size,
    deconv_output_strides,
    deconv_output_channels,

    kernel_sizes,
    strides,
    paddings,
    n_channels,
    batch_norm_deconv=False,
    batch_norm_fc=False,
    init_w=1e-3,
    hidden_init=nn.init.xavier_uniform_,
    hidden_activation=nn.ReLU()
    ):
        super(DCNNDecoder, self).__init__()
        assert (len(n_channels) == \
                len(paddings) == \
                len(strides) == \
                len(kernel_sizes)), "size of kernel, n_channels, strides, and padding is not equal"

        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_width*self.deconv_input_height*self.deconv_input_channels
        self.batch_norm_deconv = batch_norm_deconv
        self.batch_norm_fc = batch_norm_fc

        self.deconv_layers = nn.ModuleList()
        self.deconv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()
        input_size = fc_input_size

        for idx, hidden_size in enumerate(self.hidden_sizes):

            fc_layer = nn.Linear(input_size, hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)
            input_size = hidden_size
            self.fc_layers.append(fc_layer)

            fc_norm_layer = nn.BatchNorm1d(hidden_size)
            self.fc_norm_layers.append(fc_norm_layer)

        self.last_fc = nn.Linear(input_size, deconv_input_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        for out_channel, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            deconv_layer = nn.ConvTranspose2d(deconv_input_channels,
                                        out_channel,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding)
            hidden_init(deconv_layer.weight)
            deconv_layer.bias.data.fill_(0)

            self.deconv_layers.append(deconv_layer)
            deconv_input_channels = out_channel

        # a test matrix to compute number channels of CNN for BatchNorm2d
        test_mat = torch.zeros(1, self.deconv_input_channels,
                                self.deconv_input_width,
                                self.deconv_input_height)

        for deconv in self.deconv_layers:
            test_mat = deconv(test_mat)
            self.deconv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
        
        self.first_deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.first_deconv_output.weight)
        self.first_deconv_output.bias.data.fill_(0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input):
        h = self.apply_forward(input,self.fc_layers,
                                self.fc_norm_layers, use_batch_norm=self.batch_norm_fc)
        h = self.hidden_activation(self.last_fc(h))
        h = h.view(-1, self.deconv_input_channels, self.deconv_input_width,
                    self.deconv_input_height)
        h = self.apply_forward(h, self.deconv_layers, self.deconv_norm_layers,
                                use_batch_norm=self.batch_norm_deconv)
        h = self.first_deconv_output(h) 
        output = self.relu(h)
        # output = self.hidden_activation(h) 
        # output = self.sigmoid(h)
        return output

    def apply_forward(self, input, hidden_layers, norm_layers,
                        use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h

class Model(nn.Module):
    def __init__(self, encoder_param, decoder_param):
        super(Model, self).__init__()
        # print('encoder_param: ',encoder_param)
        self.encoder_param = encoder_param
        self.decoder_param = decoder_param
        self.encoder = CNNEncoder(
            input_width = self.encoder_param['input_width'],
            input_height = self.encoder_param['input_height'],
            input_channels = self.encoder_param['input_channels'],
            output_size = self.encoder_param['output_size'],
            kernel_sizes = self.encoder_param['kernel_sizes'],
            n_channels = self.encoder_param['n_channels'],
            strides = self.encoder_param['strides'],
            paddings = self.encoder_param['paddings'],
            hidden_sizes = self.encoder_param['hidden_sizes'],
            batch_norm_conv = self.encoder_param['batch_norm_conv'],
            batch_norm_fc = self.encoder_param['batch_norm_fc'],
            init_w = self.encoder_param['init_w'],
            hidden_init = self.encoder_param['hidden_init'],
            hidden_activation = self.encoder_param['hidden_activation']
        )
        self.decoder = DCNNDecoder(
            fc_input_size = self.decoder_param['fc_input_size'],
            hidden_sizes = self.decoder_param['hidden_sizes'],
            deconv_input_width = self.decoder_param['deconv_input_width'],
            deconv_input_height = self.decoder_param['deconv_input_height'],
            deconv_input_channels = self.decoder_param['deconv_input_channels'],
            kernel_sizes = self.decoder_param['kernel_sizes'],
            strides = self.decoder_param['strides'],
            paddings = self.decoder_param['paddings'],
            n_channels = self.decoder_param['n_channels'],
            batch_norm_deconv = self.decoder_param['batch_norm_deconv'],
            batch_norm_fc = self.decoder_param['batch_norm_fc'],
            init_w = self.decoder_param['init_w'],
            hidden_init = self.decoder_param['hidden_init'],
            hidden_activation = self.decoder_param['hidden_activation'],
            deconv_output_channels = self.decoder_param['deconv_output_channels'],
            deconv_output_kernel_size = self.decoder_param['deconv_output_kernel_size'],
            deconv_output_strides = self.decoder_param['deconv_output_strides']
        )
    def forward(self, input):
        z = self.encoder(input)
        input_reconstruction = self.decoder(z)
        return input_reconstruction, z
