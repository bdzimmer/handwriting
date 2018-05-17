# -*- coding: utf-8 -*-
"""

Experimental convolutional neural net based on PyTorch tutorial.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

import time

import numpy as np
import sklearn

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from handwriting import util

VERBOSE = True
VERBOSE_RATE = 100


class ExperimentalCNN(nn.Module):
    """Convnet implemented with PyTorch."""

    def __init__(
            self, input_shape, n_classes,
            conv0_kernel_hwidth=4,
            conv1_kernel_hwidth=2,
            conv0_output_channels=16,
            conv1_output_channels=16,
            linear0_size=256,
            linear1_size=128):
        """init function"""

        super().__init__()

        conv0_input_height, conv0_input_width = input_shape

        # conv0_kernel_hwidth = 4  # 4 # 4  # 2
        # conv1_kernel_hwidth = 2  # 4 # 2  # 2
        # conv0_output_channels = 16  # 4 # 2  # 4
        # conv1_output_channels = 16  # 4 # 2  # 8
        # linear0_size = 256  # 64 # 256
        # linear1_size = 128

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

        # no max pool
        # conv1_input_height = conv0_output_height
        # conv1_input_width = conv0_output_width

        self.conv1 = nn.Conv2d(
            conv0_output_channels,
            conv1_output_channels,
            conv1_kernel_hwidth * 2 + 1)
        conv1_output_height = conv1_input_height - 2 * conv1_kernel_hwidth
        conv1_output_width = conv1_input_width - 2 * conv1_kernel_hwidth

        # max pool
        self.fc0_input_size = (conv1_output_height // 2) * (conv1_output_width // 2) * conv1_output_channels

        # no max pool
        # self.fc0_input_size = conv1_output_height * conv1_output_width * conv1_output_channels
        # self.fc0_input_size = conv2_output_height * conv2_output_width * conv2_output_channels

        self.fc0 = nn.Linear(self.fc0_input_size, linear0_size)
        self.fc1 = nn.Linear(linear0_size, linear1_size)
        self.fc2 = nn.Linear(linear1_size, n_classes)

    def forward(self, x):
        """feed forward"""
        x = F.max_pool2d(F.relu(self.conv0(x)), 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        # x = F.relu(self.conv0(x))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))

        x = x.view(-1, self.fc0_input_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImagesDataset(Dataset):
    """Images dataset."""

    def __init__(self, images, labels):
        self.data = list(zip(images, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ImagesDatasetLazy(Dataset):
    """Images dataset."""

    def __init__(self, images, labels, extractor):
        self.data = list(zip(images, labels))
        self.extractor = extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return self.extractor(img), label


class CallableTorchModel(object):
    """Wrapper for torch models to make them callable."""

    def __init__(self, model, unique_labels):
        """init method"""
        self.model = model
        self.unique_labels = unique_labels

    def __call__(self, img):
        """model becomes callable"""
        img_var = Variable(
            torch.from_numpy(img[None, None, :, :]), requires_grad=True)
        output = self.model(img_var)
        _, label_idx = torch.max(output.data, 1)
        return self.unique_labels[label_idx[0]]

    def predict_proba(self, img):
        """helper"""
        # TODO: probably name this something else
        # Eventually, I would rather achieve dynamic dispatch via polymorphic wrapper
        # functions rather than methods on classes
        img = Variable(
            torch.from_numpy(img[None, None, :, :]), requires_grad=True)
        output = self.model(img)
        proba = nn.Softmax(dim=1)(output).data
        return proba


def experimental_cnn(
        nn_arch,
        nn_opt,
        epoch_log_filename,
        callback_log_filename,
        callback,
        callback_rate,
        lazy_extractor,
        save_model_filename,
        tsv_filename):

    """Build a function that fits a CNN."""

    def fit(feats_train, labels_train):
        """Perform fitting."""

        # assumes that all images are the same size and that at least
        # one training sample is passed in
        input_shape = feats_train[0].shape

        unique_labels = sorted(list(set(labels_train)))
        label_map = {k: v for v, k in enumerate(unique_labels)}
        n_classes = len(unique_labels)

        net = ExperimentalCNN(input_shape, n_classes, **nn_arch)

        optimizer = optim.SGD(
            net.parameters(),
            lr=nn_opt.get("learning_rate", 0.001),
            momentum=nn_opt.get("momentum", 0.9),
            weight_decay=nn_opt.get("weight_decay", 0.005))

        loss_func = nn.CrossEntropyLoss()

        # open log files

        epoch_log_file = (
            open(epoch_log_filename, "w")
            if epoch_log_filename is not None
            else None)

        callbacks_log_file = (
            open(callback_log_filename, "w")
            if callback_log_filename is not None
            else None)

        # prepare data loader
        batch_size = nn_opt.get("batch_size", 16)
        if lazy_extractor is None:
            dataloader = DataLoader(
                ImagesDataset(feats_train, labels_train),
                batch_size=batch_size,
                shuffle=True)
        else:
            dataloader = DataLoader(
                ImagesDatasetLazy(feats_train, labels_train, lazy_extractor),
                batch_size=batch_size,
                shuffle=True)

        # main training loop

        start_time = time.time()

        max_epochs = nn_opt.get("max_epochs", 16)
        for epoch in range(max_epochs):

            epoch_loss = 0.0
            epoch_grad_magnitude = 0.0

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

                if VERBOSE:
                    if idx % VERBOSE_RATE == 0:
                        # minibatch losses are averaged rather than summed
                        print(
                             idx * batch_size,
                             np.round(epoch_loss / (idx + 1), 6))

            mean_loss = epoch_loss / (idx + 1)
            mean_grad_magnitude = epoch_grad_magnitude / (idx + 1)

            running_time = time.time() - start_time

            status = {}

            if epoch_log_file is not None:
                model = CallableTorchModel(net, unique_labels)

                if lazy_extractor is None:
                    labels_train_pred = [model(x) for x in feats_train]
                else:
                    labels_train_pred = [model(lazy_extractor(x)) for x in feats_train]
                # TODO: something generic here instead of sklearn
                train_accuracy = sklearn.metrics.accuracy_score(
                    labels_train, labels_train_pred)

                log_line = ", ".join(
                    [str(x) for x in [
                        epoch,
                        mean_loss,
                        mean_grad_magnitude,
                        train_accuracy,
                        running_time]])
                print(log_line)
                print(log_line, file=epoch_log_file, flush=True)

                status["epoch"] = epoch
                status["mean_loss"] = mean_loss
                status["mean_grad_magnitude"] = mean_grad_magnitude
                status["train_accuracy"] = train_accuracy
                status["running_time"] = running_time

            if callbacks_log_file is not None and (epoch + 1) % callback_rate == 0:
                model = CallableTorchModel(net, unique_labels)
                callback_results = callback(model)
                callback_log_line = ", ".join(
                    [str(x[1]) for x in (
                        [("epoch", epoch)] + callback_results)])
                print(callback_log_line)
                print(callback_log_line, file=callbacks_log_file, flush=True)
                for key, val in callback_results:
                    status[key] = val

            if save_model_filename is not None:
                model = CallableTorchModel(net, unique_labels)
                util.save(model, save_model_filename + "." + format(epoch, "03d"))

            # save tsv file of status
            if tsv_filename is not None:
                with open(tsv_filename + "." + format(epoch, "03d"), "w") as tsv_file:
                    for key, val in status.items():
                        print(key + "\t" + str(val), file=tsv_file)

            print(
                "epoch", epoch, "/", max_epochs - 1, ":",
                np.round(mean_loss, 6),
                np.round(running_time, 2), "sec")
            # estimate total time to complete max_epochs
            frac_complete = (epoch + 1.0) / max_epochs
            total_time_est = running_time / frac_complete
            print(
                "ETA:",
                time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(start_time + total_time_est)),
                "(" + str(np.round((1.0 - frac_complete) * total_time_est / 60, 2)) + " min remaining; " +
                str(np.round(total_time_est / 60, 2)) + " min total)")

        # close open log files
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
    for param in model.parameters():
        if param.grad is not None:
            res += torch.sum(torch.abs(param.grad.data))

    return res
