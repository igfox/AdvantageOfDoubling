from collections import OrderedDict
import torch
from torch import nn


class NothingButNetDQN(nn.Module):
    """
    Dueling Deep Q Network with architecture described in
    The Advantage of Doubling: A Deep Reinforcement Learning Approach to Studying the Double Team in the NBA
    http://www.sloansportsconference.com/wp-content/uploads/2018/02/2010.pdf

    Expected input is a processed form of the SportsVU player tracking data. The required processing is described in
    the above paper.
    """
    def __init__(self, max_advantage=False):
        """
        :param max_advantage: boolean switch to use proper defintion of advantage (max as opposed to mean)
        """
        super(NothingButNetDQN, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv2d_0', nn.Conv2d(17, 32, kernel_size=3)),
                ('maxpool_0', nn.MaxPool2d(kernel_size=2)),
                ('relu_0', nn.ReLU(inplace=True)),
                ('batchnorm2d_0', nn.BatchNorm2d(32)),
                ('dropout2d_0', nn.Dropout2d()),

                ('conv2d_1', nn.Conv2d(32, 32, kernel_size=3)),
                ('maxpool_1', nn.MaxPool2d(kernel_size=2)),
                ('relu_1', nn.ReLU(inplace=True)),
                ('batchnorm2d_1', nn.BatchNorm2d(32)),
                ('dropout2d_1', nn.Dropout2d()),

                ('conv2d_2', nn.Conv2d(32, 32, kernel_size=3)),
                ('maxpool_2', nn.MaxPool2d(kernel_size=2)),
                ('relu_2', nn.ReLU(inplace=True)),
                ('batchnorm2d_2', nn.BatchNorm2d(32)),
            ])
        )

        self.flatten = nn.Sequential(
            OrderedDict([
                ('fc_0', nn.Linear(32 * 4 * 4, 400)),
                ('relu_0', nn.ReLU(inplace=True)),
                ('batchnorm_0', nn.BatchNorm1d(400)),
                ('dropout_0', nn.Dropout())
            ])
        )

        flat_feature_size = 93
        self.combiner = nn.Sequential(
            OrderedDict([
                ('fc_0', nn.Linear(400 + flat_feature_size, 200)),
                ('relu_0', nn.ReLU(inplace=True)),
                ('batchnorm_0', nn.BatchNorm1d(200)),
                ('dropout_0', nn.Dropout()),
                ('fc_1', nn.Linear(200, 100)),
                ('relu_1', nn.ReLU(inplace=True)),
                ('batchnorm_1', nn.BatchNorm1d(100)),
                ('dropout_1', nn.Dropout())

            ])
        )

        self.value = nn.Sequential(
            OrderedDict([
                ('fc_0', nn.Linear(100, 50)),
                ('relu_0', nn.ReLU(inplace=True)),
                ('batchnorm_0', nn.BatchNorm1d(50)),
                ('fc_1', nn.Linear(50, 1))
            ])
        )

        self.advantage = nn.Sequential(
            OrderedDict([
                ('fc_0', nn.Linear(100, 50)),
                ('relu_0', nn.ReLU(inplace=True)),
                ('batchnorm_0', nn.BatchNorm1d(50)),
                ('fc_1', nn.Linear(50, 20))
            ])
        )

        self.to_init = ['features.conv2d_0',
                        'features.conv2d_1',
                        'features.conv2d_2',
                        'flatten.fc_0',
                        'combiner.fc_0',
                        'combiner.fc_1',
                        'value.fc_0',
                        'value.fc_1',
                        'advantage.fc_0',
                        'advantage.fc_1']
        self.max_advantage = max_advantage

    def get_final(self, x):
        x, x_flat = x
        feat = self.features(x)
        flat = self.flatten(feat.view(feat.size(0), 32 * 4 * 4))
        combined = torch.cat((flat, x_flat), 1)
        fused_feat = self.combiner(combined)
        return fused_feat

    def forward(self, x):
        final = self.get_final(x)
        value = self.value(final)
        advantage = self.advantage(final)
        if self.max_advantage:
            max_advantage, maxind_advantage = torch.max(advantage, 1)
            q = value + (advantage - max_advantage.view(-1, 1))
        else:
            mean_advantage = torch.mean(advantage, 1)
            # Note: this is not the mathematical definition of advantage, but mirrors the implementation of
            # https://arxiv.org/abs/1511.06581
            q = value + (advantage - mean_advantage.view(-1, 1))
        return q

    def get_advantage(self, x):
        final = self.get_final(x)
        advantage = self.advantage(final)
        if self.max_advantage:
            max_advantage, maxind_advantage = torch.max(advantage, 1)
            return advantage - max_advantage.view(-1, 1)
        else:
            mean_advantage = torch.mean(advantage, 1)
            return advantage - mean_advantage.view(-1, 1)

    def get_value(self, x):
        final = self.get_final(x)
        value = self.value(final)
        return value
