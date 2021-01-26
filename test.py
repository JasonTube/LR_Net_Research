import os
import torch
import numpy as np
from tqdm import tqdm


class Tester:
    def __init__(self, args):
        self.device = args.device
        self.model = args.model
        model_name = self.model.__class__.__name__
        model_path = os.path.join('checkpoints',
                                  model_name,
                                  'best_model.pth.tar')
        best_model = torch.load(model_path)
        self.model = best_model['state_dict']
        self.test_loader = args.test_loader

    def one_hot_label(self, Y, n):
        y = torch.zeros([len(Y), n], device=self.device)
        y[torch.arange(0, len(Y), device=self.device), Y] = 1
        return y

    def test(self):
        amount_num = 0.
        correct_num = 0.
        batch_num = 0.

        for batch, [inputs, labels] in enumerate(tqdm(self.test_loader)):
            batch_num += 1
            inputs = inputs.reshape(-1, 1, 28, 28).to(self.device)
            labels = self.one_hot_label(labels, 10).to(self.device)

            outputs = self.model.forward(inputs)

            _outputs = torch.argmax(outputs, dim=-1).cpu().numpy()
            _labels = torch.argmax(labels, dim=-1).cpu().numpy()
            amount_num += len(inputs)
            correct_num += np.sum((_labels == _outputs))

        accuracy = correct_num / amount_num

        print('accuracy on the test set is:{}'.format(accuracy))
