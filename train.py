import os
import torch
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from optim import LrScheduler, SGD, AdamGD


def save_model(state, is_best=None, save_dir=None):
    last_model = os.path.join(save_dir, 'last_model.pth.tar')
    torch.save(state, last_model)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(last_model, best_model)


def repeat_data(data, repeat):
    data = data.reshape(len(data), -1)
    repeated_data = torch.cat([data] * repeat, dim=0)
    return repeated_data


class Trainer:

    def __init__(self, args):
        self.device = args.device
        self.draw_loss = args.draw_loss
        self.repeat_num = args.repeat_num

        self.model = args.model
        self.model_name = self.model.__class__.__name__
        self.model_path = self._model_path()

        self.epochs_SGD = args.epochs_SGD
        self.epochs_AdamGD = args.epochs_AdamGD
        self.lr = args.lr
        self.optimizer_SGD = SGD(lr=self.lr, params=self.model.get_params(), device=self.device)
        self.optimizer_AdamGD = AdamGD(lr=self.lr, params=self.model.get_params(), device=self.device)
        self.lr_scheduler = LrScheduler(args.step_size, args.gamma)

        self.train_loader = args.train_loader
        self.valid_loader = args.valid_loader

    def _model_path(self):
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        path = os.path.join('checkpoints', self.model_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def one_hot_label(self, label, label_num):
        one_hot = torch.zeros([len(label), label_num], device=self.device)
        one_hot[torch.arange(0, len(label), device=self.device), label] = 1
        return one_hot

    def valid(self, epoch):
        valid_loss = 0.
        amount_num = 0.
        correct_num = 0.
        batch_num = 0.

        for batch, [inputs, labels] in enumerate(tqdm(self.valid_loader)):
            batch_num += 1
            inputs = inputs.reshape(-1, 1, 28, 28).to(self.device)
            labels = self.one_hot_label(labels, 10).to(self.device)

            outputs = self.model.forward(inputs)
            valid_loss += torch.mean(-(labels * torch.log(outputs)).sum(dim=-1)).cpu().numpy()

            _outputs = torch.argmax(outputs, dim=-1).cpu().numpy()
            _labels = torch.argmax(labels, dim=-1).cpu().numpy()
            amount_num += len(inputs)
            correct_num += np.sum((_labels == _outputs))

        valid_loss /= batch_num
        accuracy = correct_num / amount_num

        print('valid epoch:{} loss:{} acc:{}'.format(epoch+1, valid_loss, accuracy))
        return valid_loss

    def train(self):
        print('Start training ...')

        train_losses = []
        valid_losses = []

        best_loss = 1.e10

        for epoch in range(self.epochs_SGD + self.epochs_AdamGD):
            batch_num = 0.
            train_loss = 0.
            for batch, [inputs, labels] in enumerate(tqdm(self.train_loader)):
                batch_num += 1
                inputs = repeat_data(inputs, self.repeat_num).reshape(-1, 1, 28, 28).to(self.device)
                labels = self.one_hot_label(labels, 10).to(self.device)
                labels = repeat_data(labels, self.repeat_num).to(self.device)

                outputs = self.model.forward(inputs)
                loss = -torch.sum(labels * torch.log(outputs), dim=-1)
                grads = self.model.backward(loss)

                if epoch < self.epochs_SGD:
                    self.optimizer_SGD.update_lr(self.lr)
                    params = self.optimizer_SGD.update_params(grads)
                else:
                    self.optimizer_AdamGD.update_lr(self.lr)
                    params = self.optimizer_AdamGD.update_params(grads)
                self.model.set_params(params)

                train_loss += torch.mean(loss).cpu().numpy()

            train_loss /= batch_num
            train_losses += [train_loss]
            print('train epoch:{} loss:{} learning rate:{}'.format(epoch+1, train_loss, self.lr))
            self.lr = self.lr_scheduler.step(self.lr)

            valid_loss = self.valid(epoch)
            valid_losses += [valid_loss]
            is_best = valid_loss < best_loss
            best_loss = valid_loss if is_best else best_loss
            state = {'epoch': epoch,
                     'state_dict': self.model,
                     'best_loss': best_loss}
            save_model(state, is_best, save_dir=self.model_path)

        print('Finished training ...')
        np.save('loss.npy', [train_losses, valid_losses])

        if self.draw_loss:
            loss_data = np.load('loss.npy', allow_pickle=True)
            train_losses = loss_data[0]
            valid_losses = loss_data[1]
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            ax.set_title('Loss curve', usetex=True, fontsize=20)
            ax.set_xlabel('batch', usetex=True, fontsize=20)
            ax.set_ylabel('loss', usetex=True, fontsize=20)
            ax.plot([x for x in range(1, len(train_losses)+1)], train_losses, color='g', label='train loss')
            ax.plot([x for x in range(1, len(valid_losses)+1)], valid_losses, color='r', label='valid loss')
            ax.legend(frameon=False, loc='best')
            plt.show()

