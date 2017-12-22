# -*- coding: utf-8 -*-
"""

Experimental convolutional neural net based on PyTorch tutorial.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import time

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable


class ExperimentalCNN(nn.Module):
    """Convnet implemented with PyTorch."""

    def __init__(self, n_classes):
        """init function"""
        super().__init__()

        # input is a 96 * 96 grayscale image

        # conv2d args are in channels, out channels, kernel size
        self.conv0 = nn.Conv2d(1, 4, 5)
        # max pool
        self.conv1 = nn.Conv2d(4, 8, 5)
        # max pool
        self.fc0 = nn.Linear(3528, 256)
        self.fc1 = nn.Linear(256, n_classes)

    def forward(self, x):
        """feed forward"""
        x = F.max_pool2d(F.relu(self.conv0(x)), 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = x.view(-1, _num_flat_features(x))
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return x


def experimental_cnn(max_epochs):
    """build a function that fits a CNN"""

    def fit(feats_train, labels_train):
        """Perform fitting."""

        unique_labels = sorted(list(set(labels_train)))
        label_map = {k: v for v, k in enumerate(unique_labels)}
        n_classes = len(unique_labels)

        net = ExperimentalCNN(n_classes)
        # TODO: expose optimizer params
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        loss_func = nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(max_epochs):

            idxs_shuffled = np.random.permutation(len(labels_train))
            epoch_loss = 0.0

            for idx, idx_shuffled in enumerate(idxs_shuffled):

                # something in here is resetting the threads
                # torch.set_num_threads(2)

                img = np.array(feats_train[idx_shuffled], dtype=np.float32)
                img = Variable(torch.from_numpy(img[None, None, :, :]), requires_grad=True)
                label = labels_train[idx_shuffled]

                # this isn't necessary
                # target = np.zeros(n_classes, dtype=np.float32)
                # target[label_map[label]] = 1.0
                # target = Variable(torch.from_numpy(target))
                target = Variable(torch.LongTensor([label_map[label]]))

                # zero the gradient buffers
                optimizer.zero_grad()
                # feedforward
                output = net(img)
                loss = loss_func(output, target)
                # backpropagate
                loss.backward()
                optimizer.step()

                epoch_loss += loss.data[0]

                if idx % 100 == 0:
                    print(idx, np.round(epoch_loss / (idx + 1), 6))

            total_time = time.time() - start_time
            print("epoch", epoch, ":", np.round(epoch_loss / (idx + 1), 6), np.round(total_time), "sec")

        def predict(feats_test):
            """make predictions using the fitted model"""
            res = []
            for img in feats_test:
                img = np.array(img, dtype=np.float32)
                img = Variable(torch.from_numpy(img[None, None, :, :]), requires_grad=True)
                output = net(img)
                _, label_idx = torch.max(output.data, 1)
                res.append(unique_labels[label_idx[0]])
            return res

        return predict

    return fit


def _num_flat_features(x):
    """find size of x except for the batch dimension"""
    # TODO: better way to do this
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
