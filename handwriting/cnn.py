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

    def __init__(self, input_shape, n_classes):
        """init function"""
        super().__init__()

        conv0_input_height, conv0_input_width = input_shape

        conv0_kernel_hwidth = 2
        conv1_kernel_hwidth = 2

        conv0_output_channels = 4
        conv1_output_channels = 8

        linear_size = 256

        # conv2d args are in channels, out channels, kernel size
        self.conv0 = nn.Conv2d(
            1,
            conv0_output_channels,
            conv0_kernel_hwidth * 2 + 1)

        conv0_output_height = conv0_input_height - 2 * conv0_kernel_hwidth
        conv0_output_width = conv0_input_width - 2 * conv0_kernel_hwidth

        # max pool
        conv1_input_height = conv0_output_height // 2
        conv1_input_width = conv0_output_width // 2

        self.conv1 = nn.Conv2d(
            conv0_output_channels,
            conv1_output_channels,
            conv1_kernel_hwidth * 2 + 1)

        conv1_output_height = conv1_input_height - 2 * conv1_kernel_hwidth
        conv1_output_width = conv1_input_width - 2 * conv1_kernel_hwidth

        # max pool
        self.fc0_input_size = (conv1_output_height // 2) * (conv1_output_width // 2) * conv1_output_channels

        self.fc0 = nn.Linear(self.fc0_input_size, linear_size)
        self.fc1 = nn.Linear(linear_size, n_classes)

    def forward(self, x):
        """feed forward"""
        x = F.max_pool2d(F.relu(self.conv0(x)), 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = x.view(-1, self.fc0_input_size)
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


class CallableTorchModel(object):
    """Wrapper for torch models to make them callable."""

    def __init__(self, model, unique_labels):
        """init method"""
        self.model = model
        self.unique_labels = unique_labels

    def __call__(self, feats):
        """model becomes callable"""
        # not sure that this generalizes, but will use it for now
        res = []
        for img in feats:
            # img = np.array(img, dtype=np.float32)
            img = Variable(
                torch.from_numpy(img[None, None, :, :]), requires_grad=True)
            output = self.model(img)
            _, label_idx = torch.max(output.data, 1)
            res.append(self.unique_labels[label_idx[0]])
        return res

    def predict_proba(self, feats):
        """helper"""
        # TODO: probably name this something else
        # I would rather achieve dynamic dispatch via polymorphic wrapper
        # functions rather than methods on classes

        res = []
        for img in feats:
            # img = np.array(img, dtype=np.float32)
            img = Variable(
                torch.from_numpy(img[None, None, :, :]), requires_grad=True)
            output = self.model(img)
            proba = nn.Softmax(dim=1)(output).data
            # print("labels:", self.unique_labels)
            # print("output:", output)
            # print("proba:", proba)
            res.append(proba)
        return res


def experimental_cnn(
        batch_size,
        max_epochs,
        learning_rate,
        momentum,
        epoch_log_filename,
        callback_log_filename,
        callback,
        callback_rate):

    """Build a function that fits a CNN."""

    def fit(feats_train, labels_train):
        """Perform fitting."""

        unique_labels = sorted(list(set(labels_train)))
        label_map = {k: v for v, k in enumerate(unique_labels)}
        n_classes = len(unique_labels)
        n_samples = len(feats_train)

        # assumes that all images are the same size and that at least
        # one training sample is passed in
        input_shape = feats_train[0].shape

        net = ExperimentalCNN(input_shape, n_classes)

        # in the example, learning rate was 0.001 and momentum was 0.9
        optimizer = optim.SGD(
            net.parameters(), lr=learning_rate, momentum=momentum)
        loss_func = nn.CrossEntropyLoss()

        epoch_log_file = (
            open(epoch_log_filename, "w")
            if epoch_log_filename is not None
            else None)

        callbacks_log_file = (
            open(callback_log_filename, "w")
            if callback_log_filename is not None
            else None)

        start_time = time.time()

        for epoch in range(max_epochs):

            epoch_loss = 0.0
            epoch_grad_magnitude = 0.0

            if batch_size == 0:

                # TODO: get rid of this

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
                    time.localtime(start_time + total_time_est)),
                "(" + str(np.round(total_time_est, 2)) + " sec)")

            if epoch_log_file is not None:
                model = CallableTorchModel(net, unique_labels)

                labels_train_pred = model(feats_train)
                # TODO: something generic here instead of sklearn
                import sklearn
                accuracy = sklearn.metrics.accuracy_score(
                    labels_train, labels_train_pred)

                print(
                    ", ".join(
                        [str(x) for x in [
                            epoch,
                            mean_loss,
                            mean_grad_magnitude,
                            accuracy,
                            running_time]]),
                    file=epoch_log_file,
                    flush=True)

            if callbacks_log_file is not None and epoch % callback_rate == 0:
                model = CallableTorchModel(net, unique_labels)
                callback_results = callback(model)
                print(
                    ", ".join(
                        [str(x) for x in (
                            [epoch] + callback_results)]),
                    file=callbacks_log_file,
                    flush=True)

        if epoch_log_file is not None:
            epoch_log_file.close()

        if callbacks_log_file is not None:
            callbacks_log_file.close()

        return CallableTorchModel(net, unique_labels)

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
