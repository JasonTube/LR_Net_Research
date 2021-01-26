import torch


class LrScheduler:

    def __init__(self, step_size, gamma):
        self.epoch = 0
        self.step_size = step_size
        self.gamma = gamma

    def step(self, lr):
        self.epoch += 1
        if (self.epoch % self.step_size) == 0:
            return lr * self.gamma
        else:
            return lr


class Optimizer:
    def __init__(self, lr, params, device=torch.device("cuda:0")):
        self.lr = lr
        self.params = params
        self.device = device

    def update_params(self, grads):
        pass

    def update_lr(self, lr):
        self.lr = lr


class SGD(Optimizer):

    def __init__(self, lr, params, device=torch.device("cuda:0")):
        super().__init__(lr, params, device)

    def update_params(self, grads):
        for key in self.params:
            self.params[key] = self.params[key] - self.lr * grads['grad_' + key]
        return self.params


class AdamGD(Optimizer):

    def __init__(self, lr, params, beta1=0.9, beta2=0.999, epsilon=1e-8, device=torch.device("cuda:0")):
        super().__init__(lr, params, device)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum['vd' + key] = torch.zeros(self.params[key].shape, device=self.device)
            self.rmsprop['sd' + key] = torch.zeros(self.params[key].shape, device=self.device)

    def update_params(self, grads):

        for key in self.params:
            # Momentum update.
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads['grad_' + key]
            # RMSprop update.
            self.rmsprop['sd' + key] = (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * (
                    grads['grad_' + key] ** 2)
            # Update parameters.
            self.params[key] = self.params[key] - (self.lr * self.momentum['vd' + key]) / (
                    torch.sqrt(self.rmsprop['sd' + key]) + self.epsilon)

        return self.params
