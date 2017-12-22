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


def experimental_cnn(
        max_epochs,
        learning_rate,
        momentum,
        log_filename=None):

    """Build a function that fits a CNN."""

    def fit(feats_train, labels_train):
        """Perform fitting."""

        unique_labels = sorted(list(set(labels_train)))
        label_map = {k: v for v, k in enumerate(unique_labels)}
        n_classes = len(unique_labels)

        net = ExperimentalCNN(n_classes)
        # in the example, learning rate was 0.001 and momentum was 0.9
        optimizer = optim.SGD(
            net.parameters(), lr=learning_rate, momentum=momentum)
        loss_func = nn.CrossEntropyLoss()

        log_file = open(log_filename, "w") if log_filename is not None else None
        start_time = time.time()

        for epoch in range(max_epochs):

            idxs_shuffled = np.random.permutation(len(labels_train))
            epoch_loss = 0.0

            epoch_grad_magnitude = 0.0

            # TODO: figure out how to use minibatches properly
            for idx, idx_shuffled in enumerate(idxs_shuffled):

                # something in here is resetting the thread count
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

                # TODO: how do I look at the size of the gradient?
                # this is how SGD does it
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            # not sure which one of these is more appropriate
                            # epoch_grad_magnitude += torch.norm(p.grad.data)
                            epoch_grad_magnitude += torch.sum(torch.abs(p.grad.data))

                if idx % 100 == 0:
                    print(idx, np.round(epoch_loss / (idx + 1), 6))

            mean_loss = epoch_loss / (idx + 1)
            mean_grad_magnitude = epoch_grad_magnitude / (idx + 1)

            total_time = time.time() - start_time
            print(
                "epoch", epoch, ":",
                np.round(mean_loss),
                np.round(total_time), "sec")
            if log_file is not None:
                print(", ".join(
                    [str(x) for x in [
                        epoch,
                        mean_loss,
                        mean_grad_magnitude,
                        total_time]]),
                    file=log_file, flush=True)

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
