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
from torch.utils.data import Dataset, DataLoader


class ExperimentalCNN(nn.Module):
    """Convnet implemented with PyTorch."""

    def __init__(self, n_classes):
        """init function"""
        super().__init__()

        # TODO: calculate these sizes programmatically
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
        x = x.view(-1, 3528)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return x


class ImagesDataset(Dataset):
    """Images dataset."""

    def __init__(self, images, labels):
        self.data = list(zip(images, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def experimental_cnn(
        batch_size,
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
        n_samples = len(feats_train)

        net = ExperimentalCNN(n_classes)
        # in the example, learning rate was 0.001 and momentum was 0.9
        optimizer = optim.SGD(
            net.parameters(), lr=learning_rate, momentum=momentum)
        loss_func = nn.CrossEntropyLoss()

        log_file = open(log_filename, "w") if log_filename is not None else None
        start_time = time.time()

        for epoch in range(max_epochs):

            epoch_loss = 0.0
            epoch_grad_magnitude = 0.0

            if batch_size == 0:

                idxs_shuffled = np.random.permutation(n_samples)
                for idx, idx_shuffled in enumerate(idxs_shuffled):
                    img = np.array(feats_train[idx_shuffled], dtype=np.float32)
                    img = Variable(torch.from_numpy(img[None, None, :, :]), requires_grad=True)
                    label = labels_train[idx_shuffled]
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
                    epoch_grad_magnitude += grad_magnitude(optimizer)

                    if idx % 100 == 0:
                        print(idx, np.round(epoch_loss / (idx + 1), 6))

                mean_loss = epoch_loss / n_samples
                mean_grad_magnitude = epoch_grad_magnitude / n_samples

            else:

                dataloader = DataLoader(
                    ImagesDataset(feats_train, labels_train),
                    batch_size=batch_size,
                    shuffle=True)

                for idx, (images, labels) in enumerate(dataloader):
                    ims = Variable(
                        torch.from_numpy(np.array(
                            [np.array(image[None, :, :], dtype=np.float32)
                             for image in images])),
                        requires_grad=True)
                    targets = Variable(
                        torch.LongTensor(
                            [label_map[label] for label in labels]))
                    # zero the gradient buffers
                    optimizer.zero_grad()
                    # feedforward
                    output = net(ims)
                    loss = loss_func(output, targets)
                    # backpropagate
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.data[0]
                    epoch_grad_magnitude += grad_magnitude(net)

                    if idx % 25 == 0:
                        # minibatch losses are averaged rather than summed
                        print(
                            idx * batch_size,
                            np.round(epoch_loss / (idx + 1), 6))

                mean_loss = epoch_loss / (idx + 1)
                mean_grad_magnitude = epoch_grad_magnitude / (idx + 1)

            running_time = time.time() - start_time
            print(
                "epoch", epoch, ":",
                np.round(mean_loss, 6),
                np.round(running_time, 2), "sec")

            # estimate total time to complete max_epochs
            frac_complete = (epoch + 1.0) / max_epochs
            total_time_est = running_time / frac_complete
            print(
                "Estimated completion time:",
                time.strftime(
                    '%Y-%m-%d %H:%M:%S',
                    time.localtime(start_time + total_time_est)))

            if log_file is not None:
                print(", ".join(
                    [str(x) for x in [
                        epoch,
                        mean_loss,
                        mean_grad_magnitude,
                        running_time]]),
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


def grad_magnitude(model):
    """calculate the magnitude of the current gradient of a model"""

    # use to check for vanishing / exploding gradient

    res = 0.0

    # this is how SGD does it
    # for group in optimizer.param_groups:
    #     for p in group["params"]:
    #         if p.grad is not None:
    #             # not sure which one of these is more appropriate
    #             # res += torch.norm(p.grad.data)
    #             res += torch.sum(torch.abs(p.grad.data))

    # this is simpler and seems to produce similar results
    for param in model.parameters():
        if param.grad is not None:
            res += torch.sum(torch.abs(param.grad.data))

    return res
