"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Script loads the CIFAR-10 dataset.

@author: Sebastian Doerrich
@copyright: Copyright (c) 2022, Chair of Explainable Machine Learning (xAI), Otto-Friedrich University of Bamberg
@credits: [Christian Ledig, Sebastian Doerrich]
@license: CC BY-SA
@version: 1.0
@python: Python 3
@maintainer: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
@status: Production
"""

import pickle
import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset


class Cifar(Dataset):
    def __init__(self, path='data/cifar-10-batches-py/', transform=None, train=True, samples=None, balance=True):

        self.transform = transform
        self.cls_num_list = []
        if train:
            train_idx = [1, 2, 3, 4, 5]

            # training data
            training_data = []
            training_label = []
            for idx in train_idx:
                data_path = path + 'data_batch_' + str(idx)
                with open(data_path, 'rb') as fp:
                    dict = pickle.load(fp, encoding='bytes')
                    labels = dict[b'labels']
                    data = dict[b'data'].reshape(-1, 3, 32, 32)
                    training_data.append(data)
                    training_label.append(labels)
            self.data = np.concatenate(training_data, axis=0)
            self.data = self.data.transpose((0, 2, 3, 1))
            self.label = np.concatenate(training_label, axis=0)

            if samples is not None:
                class_labels = list(range(10))

                if balance:
                    weights = [0.1] * 10
                else:
                    weights = [0.4, 0.24, 0.14, 0.08, 0.05, 0.04, 0.03, 0.01, 0.006, 0.004]

                data_ = []
                label_ = []

                for l in class_labels:
                    label_mask = (self.label == l)
                    masked_images = self.data[label_mask, :, :, :]
                    masked_labels = self.label[label_mask]
                    num_samples_per_class = int(samples * weights[l])
                    masked_images = masked_images[:num_samples_per_class, :, :, :]
                    masked_labels = masked_labels[:num_samples_per_class]
                    data_.append(masked_images)
                    label_.append(masked_labels)
                    self.cls_num_list.append(masked_images.shape[0])

                self.data = np.concatenate(data_, axis=0)
                self.label = np.concatenate(label_, axis=0)

        else:
            with open(path + 'test_batch', 'rb') as fp:
                dict = pickle.load(fp, encoding='bytes')
                labels = dict[b'labels']
                data = dict[b'data'].reshape(-1, 3, 32, 32)
                self.data = data.transpose((0, 2, 3, 1))
                self.label = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.label[index]
        return (img, label)

    def get_img_num_per_class(self):
        return self.cls_num_list


if __name__ == '__main__':
    x = Cifar()
    data = x.get_batched_train()
