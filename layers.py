import torch
import numpy as np
from utils import im2col


class Layer:

    def __init__(self, device=torch.device('cuda:0')):
        self.device = device
        self.cache = None

    def forward(self, input):
        pass

    def backward(self, loss):
        pass

    def init(self, init_name, weight_size, mode="fan_in", act="TanH"):
        weight_size = torch.tensor(weight_size)
        fan_in = torch.prod(weight_size[:-1])
        fan_out = weight_size[-1]
        gain = 1.

        if init_name == "xavier_normal":
            std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
            return torch.randn(size=tuple(weight_size), device=self.device) * std
        elif init_name == "kaiming_normal":
            if act == "TanH":
                gain = 5 / 3
            elif act == "ReLU":
                gain = np.sqrt(2)
            fan = fan_in if mode == 'fan_in' else fan_out
            std = gain / np.sqrt(fan)
            return torch.randn(size=tuple(weight_size), device=self.device) * std


class Linear(Layer):

    def __init__(self, input_features, output_features, noise_std=1e-0,
                 act='ReLU', device=torch.device('cuda:0')):
        super().__init__(device)
        self.input_features = input_features
        self.output_features = output_features
        self.act = act

        self.w_size = (self.input_features, self.output_features)
        self.b_size = (1, self.output_features)
        self.W = {'val': self.init("kaiming_normal", self.w_size, act=self.act), 'grad': 0.}
        self.b = {'val': self.init("kaiming_normal", self.b_size, act=self.act), 'grad': 0.}

        self.noise_std = noise_std
        self.noise = None

    def forward(self, input):
        self.cache = input
        tmp = input @ self.W['val'] + self.b['val']
        self.noise = torch.randn(size=tmp.shape, device=self.device) * self.noise_std
        return tmp + self.noise

    def backward(self, loss):
        # input.shape: N, input_features
        # loss[:, np.newaxis].shape: N, 1
        # loss: A B C | A B C | A B C | ...
        input = self.cache

        w_term = input * loss[:, np.newaxis]
        w_batch_grad = torch.einsum('ni, nj->ij', w_term, self.noise) / (self.noise_std ** 2)
        self.W['grad'] = w_batch_grad / input.shape[0]

        b_term = torch.ones(size=[len(self.noise), 1], device=self.device) * loss[:, np.newaxis]
        b_batch_grad = torch.einsum('ni, nj->ij', b_term, self.noise) / (self.noise_std ** 2)
        self.b['grad'] = b_batch_grad / input.shape[0]

        return self.W['grad'], self.b['grad']


class Conv(Layer):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 noise_std=1e-0, act='ReLU', device=torch.device('cuda:0')):
        super().__init__(device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.noise_std = noise_std
        self.act = act

        self.w_size = (self.in_channels * self.kernel_size * self.kernel_size, self.out_channels)
        self.b_size = (1, self.out_channels)
        self.W = {'val': self.init("kaiming_normal", self.w_size, act=self.act), 'grad': 0.}
        self.b = {'val': self.init("kaiming_normal", self.b_size, act=self.act), 'grad': 0.}

        self.noise = None

    def forward(self, input):

        N, C, H, W = input.shape
        _C = self.out_channels
        _H = int((H + 2 * self.padding - self.kernel_size) / self.stride) + 1
        _W = int((W + 2 * self.padding - self.kernel_size) / self.stride) + 1

        input_col = im2col(input, self.kernel_size, self.kernel_size,
                           self.stride, self.padding, self.device).T

        self.cache = input, input_col
        tmp = input_col @ self.W['val'] + self.b['val']

        self.noise = torch.randn(size=tmp.shape, device=self.device) * self.noise_std
        output_col = tmp + self.noise

        output = output_col.reshape((N, _C, _H, _W))
        # output = torch.cat(torch.split(output_col, 1, dim=0)).reshape((N, _C, _H, _W)) # main problem of speed
        return output

    def backward(self, loss):
        """
        input.shape: N, in_channels, in_height, in_width
        input_col.shape: N * conv_work_num, out_channels * kernel_size * kernel_size
        loss.shape: N
        loss_col.shape: N * conv_work_num, 1
        w_term.shape: N * conv_work_num, in_channels * kernel_size * kernel_size
        noise.shape: N * conv_work_num, out_channels
        W['val'].shape: in_channels * kernel_size * kernel_size, out_channels
        """
        input, input_col = self.cache

        loss_col = loss.expand(int(input_col.shape[0] / input.shape[0]), len(loss)).T.reshape(-1, 1)

        w_term = input_col * loss_col
        w_batch_grad = (w_term.T @ self.noise) / self.noise_std
        # w_batch_grad = torch.einsum('ni, nj->ij', w_term, self.noise) / self.noise_std
        self.W['grad'] = w_batch_grad / input_col.shape[0]

        b_term = torch.ones(size=[len(self.noise), 1], device=self.device) * loss_col
        b_batch_grad = torch.einsum('ni, nj->ij', b_term, self.noise) / self.noise_std
        self.b['grad'] = b_batch_grad / input_col.shape[0]

        return self.W['grad'], self.b['grad']


class Pool(Layer):

    def __init__(self, kernel_size, padding=0, mode='Average', device=torch.device('cuda:0')):
        super().__init__(device)
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.padding = padding
        self.mode = mode

    def forward(self, input):
        self.cache = input
        if self.mode == 'Average':
            return torch.nn.functional.avg_pool2d(input, self.kernel_size, stride=self.stride, padding=self.padding)
        elif self.mode == 'Max':
            return torch.nn.functional.max_pool2d(input, self.kernel_size, stride=self.stride, padding=self.padding)


class Activation(Layer):

    def __init__(self, act_name):
        super().__init__()
        self.act_name = act_name

    def forward(self, input):
        self.cache = input
        if self.act_name == 'TanH':
            alpha = 1.7159
            return alpha * torch.tanh(input)
        elif self.act_name == 'ReLU':
            return torch.relu(input)
        elif self.act_name == 'Softmax':
            return torch.softmax(input, dim=1)
        elif self.act_name == 'Threshold':
            input[input < 0] = 0
            return torch.sign(input)
