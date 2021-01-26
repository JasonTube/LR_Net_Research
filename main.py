from train import Trainer
from test import Tester
from dataloader import get_train_valid_loader, get_test_loader
from model import DenseNet, LeNet5, ShallowConvNet
import torch
import argparse


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA or not')
        parser.add_argument('--draw_loss', action='store_true', default=True, help='disable CUDA or not')
        parser.add_argument('--batch_size', type=int, default=32, help='batchsize for dataloader')
        parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
        parser.add_argument('--step_size', type=int, default=20, help='step size in lr_scheduler for optimizer')
        parser.add_argument('--gamma', type=float, default=0.8, help='gamma in lr_scheduler for optimizer')
        parser.add_argument('--epochs_SGD', type=int, default=1, help='epochs for SGD optimizer')
        parser.add_argument('--epochs_AdamGD', type=int, default=0, help='epochs for AdamGD optimizer')
        parser.add_argument('--repeat_num', type=int, default=100, help='data repeat num')
        self.parser = parser

    def parse(self):
        arg = self.parser.parse_args(args=[])
        arg.cuda = not arg.no_cuda and torch.cuda.is_available()
        arg.device = torch.device('cuda' if arg.cuda else 'cpu')
        return arg


if __name__ == '__main__':

    args = Options().parse()

    args.model = DenseNet(args.device)
    # args.model = LeNet5(args.device)
    # rgs.model = ShallowConvNet(args.device)
    args.train_loader, args.valid_loader = get_train_valid_loader('./',
                                                                  batch_size=args.batch_size,
                                                                  random_seed=123,
                                                                  valid_ratio=0.1,
                                                                  shuffle=True)
    trainer = Trainer(args)
    trainer.train()

    args.test_loader = get_test_loader('./',
                                       batch_size=8,
                                       shuffle=False)

    tester = Tester(args)
    tester.test()