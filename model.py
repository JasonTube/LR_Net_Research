import torch
from layers import Conv, Pool, Linear, Activation


class Net:
    def __init__(self, device=torch.device('cuda:0')):
        self.device = device
        self.layers = []

    def forward(self, input):
        pass

    def backward(self, loss):
        grads = {}
        for i, layer in enumerate(self.layers):
            grads['grad_W_' + str(i + 1)], grads['grad_b_' + str(i + 1)] = layer.backward(loss)
        return grads

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params['W_' + str(i + 1)] = layer.W['val']
            params['b_' + str(i + 1)] = layer.b['val']
        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.W['val'] = params['W_' + str(i + 1)]
            layer.b['val'] = params['b_' + str(i + 1)]


class DenseNet(Net):

    def __init__(self, device=torch.device('cuda:0')):
        super().__init__(device)

        self.fc1 = Linear(28*28, 50, noise_std=1e-0, device=self.device)
        self.act1 = Activation('TanH')
        self.fc2 = Linear(50, 10, noise_std=1e-0, device=self.device)
        self.softmax = Activation('Softmax')

        self.layers = [self.fc1, self.fc2]

    def forward(self, input):
        input = input.reshape(len(input), -1)

        fc_out_1 = self.fc1.forward(input)
        act_out_1 = self.act1.forward(fc_out_1)
        fc_out_2 = self.fc2.forward(act_out_1)
        output = self.softmax.forward(fc_out_2)
        return output


class DenseNet_CNN(Net):

    def __init__(self, device=torch.device('cuda:0')):
        super().__init__(device)

        # self.fc1 = Linear(28*28, 25, noise_std=1e-0, device=self.device)
        self.fc1 = Conv(1, 25, kernel_size=25, noise_std=1e-0, act='TanH', device=self.device)
        self.act1 = Activation('TanH')
        self.fc2 = Linear(16*25, 10, noise_std=1e-0, device=self.device)
        self.softmax = Activation('Softmax')

        self.layers = [self.fc1, self.fc2]

    def forward(self, input):
        #input = input.reshape(len(input), -1)

        fc_out_1 = self.fc1.forward(input)
        act_out_1 = self.act1.forward(fc_out_1)

        act_out_1 = act_out_1.reshape(len(act_out_1), -1)

        fc_out_2 = self.fc2.forward(act_out_1)
        output = self.softmax.forward(fc_out_2)
        return output


class LeNet5(Net):

    def __init__(self, device=torch.device('cuda:0')):
        super().__init__(device)

        self.conv1 = Conv(1, 6, kernel_size=5, noise_std=1e-0, act='TanH', device=self.device)
        self.act1 = Activation('TanH')
        self.pool1 = Pool(2, device=self.device)

        self.conv2 = Conv(6, 16, kernel_size=5, noise_std=1e-0, act='TanH', device=self.device)
        self.act2 = Activation('TanH')
        self.pool2 = Pool(2, device=self.device)

        self.fc1 = Linear(256, 120, noise_std=1e-0, act='TanH', device=self.device)
        self.act3 = Activation('TanH')
        self.fc2 = Linear(120, 84, noise_std=1e-0, act='TanH', device=self.device)
        self.act4 = Activation('TanH')
        self.fc3 = Linear(84, 10, noise_std=1e-0, act='TanH', device=self.device)
        self.softmax = Activation('Softmax')

        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def forward(self, input):
        # N, 1, 28, 28
        conv_out_1 = self.conv1.forward(input)
        # N, 6, 24, 24
        act_out_1 = self.act1.forward(conv_out_1)
        pool_out_1 = self.pool1.forward(act_out_1)
        # N, 6, 12, 12

        conv_out_2 = self.conv2.forward(pool_out_1)
        # N, 16, 8, 8
        act_out_2 = self.act2.forward(conv_out_2)
        pool_out_2 = self.pool2.forward(act_out_2)
        # N, 16, 4, 4

        pool_out_2 = pool_out_2.reshape(len(pool_out_2), -1)
        # N, 256

        fc_out_1 = self.fc1.forward(pool_out_2)
        act_out_3 = self.act3.forward(fc_out_1)
        fc_out_2 = self.fc2.forward(act_out_3)
        act_out_4 = self.act4.forward(fc_out_2)
        fc_out_3 = self.fc3.forward(act_out_4)
        output = self.softmax.forward(fc_out_3)

        return output


class ShallowConvNet(Net):

    def __init__(self, device=torch.device('cuda:0')):
        super().__init__(device)

        self.conv1 = Conv(1, 6, kernel_size=5, noise_std=1e-0, act='TanH', device=self.device)
        self.act1 = Activation('TanH')
        self.pool1 = Pool(2, device=self.device)

        self.fc1 = Linear(6*12*12, 100, noise_std=1e-0, act='TanH', device=self.device)
        self.act2 = Activation('TanH')
        self.fc2 = Linear(100, 10, noise_std=1e-0, act='TanH', device=self.device)
        self.softmax = Activation('Softmax')

        self.layers = [self.conv1, self.fc1, self.fc2]

    def forward(self, input):
        conv_out_1 = self.conv1.forward(input)
        act_out_1 = self.act1.forward(conv_out_1)
        pool_out_1 = self.pool1.forward(act_out_1)

        pool_out_1 = pool_out_1.reshape(len(pool_out_1), -1)

        fc_out_1 = self.fc1.forward(pool_out_1)
        act_out_2 = self.act2.forward(fc_out_1)
        fc_out_2 = self.fc2.forward(act_out_2)
        output = self.softmax.forward(fc_out_2)

        return output
