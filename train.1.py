#!/usr/bin/env python3
import copy
import json
import time
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.INFO)

class Trainer(object):

    def __init__(self, data_dir, labels_filename):
        self.device = torch.device("cpu")
        self.labels_filename = labels_filename
        self.data_dir = data_dir
        self.train_dir = self.data_dir + "/train"
        self.valid_dir = self.data_dir + "/valid"
        self.test_dir = self.data_dir + "/test"
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.image_size = 224

        self._labels = None

    @property
    def data_transforms(self):
        return {
            "train": transforms.Compose([transforms.RandomRotation(25),
                                        transforms.RandomResizedCrop(self.image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)]),
            "valid": transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(self.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)]),
            "test" : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(self.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)])
        }

    @property
    def image_datasets(self):
        return {
            "train" : torchvision.datasets.ImageFolder(root=self.train_dir, transform=self.data_transforms["train"]),
            "valid" : torchvision.datasets.ImageFolder(root=self.valid_dir, transform=self.data_transforms["valid"]),
            "test"  : torchvision.datasets.ImageFolder(root=self.test_dir, transform=self.data_transforms["test"])
        }

    @property
    def dataloaders(self):
        return {
            "train": torch.utils.data.DataLoader(self.image_datasets["train"], batch_size=32, shuffle=True),
            "valid": torch.utils.data.DataLoader(self.image_datasets["valid"], batch_size=32),
            "test" : torch.utils.data.DataLoader(self.image_datasets["test"], batch_size=32)
        }

    @property
    def labels(self):
        if not self._labels:
            with open(self.labels_filename, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def __train_model(self, model, criterion, optimizer, num_epochs=25):
        """
        Helper function for training model
        """
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
            logging.info("-" * 10)

            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimise if only in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.image_datasets[phase])
                epoch_acc = running_corrects.double() / len(self.image_datasets[phase])

                logging.info("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == "valid" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            logging.info("")

        time_elapsed = time.time() - since
        logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        logging.info("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def train(self, arch, learning_rate, num_epochs, use_gpu, hidden_units):
        self.num_epochs = num_epochs
        self.arch = arch
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info("using device: {}".format(self.device))

        # train functions
        if not arch.startswith("vgg"):
            raise Exception("Only VGG varients are supported")

        logging.info("using pretrained model: {}".format(arch))
        model_fn = models.__dict__[arch](pretrained=True)

        # fixed feature extractor
        for param in model_fn.parameters():
            param.requires_grad = False

        output_size = len(self.labels)
        input_size = model_fn.classifier[0].in_features

        od = OrderedDict()
        hidden_sizes = hidden_units
        hidden_sizes.insert(0, input_size)

        for i in range(len(hidden_sizes) - 1):
            od['fc' + str(i + 1)] = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            od['relu' + str(i + 1)] =  nn.ReLU()
            od['dropout' + str(i + 1)] = nn.Dropout(p=0.15)
        od['output'] = nn.Linear(hidden_sizes[i + 1], output_size)
        od['softmax'] = nn.LogSoftmax(dim=1)

        self.classifier = nn.Sequential(od)

        model_fn.classifier = self.classifier

        model_fn = model_fn.to(self.device)

        self.criterion_fn = nn.NLLLoss()
        self.optimizer_fn = optim.Adam(model_fn.classifier.parameters())

        # train model
        self.model = self.__train_model(model_fn, self.criterion_fn, self.optimizer_fn,self.num_epochs)

        return self.model

    def test_accuracy(self):
        # Disabling gradient calculation
        corrects = 0
        total = 0
        total_images = len(self.dataloaders["test"].batch_sampler) * self.dataloaders["test"].batch_size

        # Disabling gradient calculation
        with torch.no_grad():
            for inputs, labels in self.dataloaders["test"]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                corrects += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = 100 * corrects // total
        logging.info("Accurately classified {:d}% of {:d} images".format(test_acc, total_images))
        return test_acc

    def save_checkpoint(self, save_dir):
        self.model.class_to_idx = self.image_datasets['train'].class_to_idx
        model_state = {
            'epoch': self.num_epochs,
            'state_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer_fn.state_dict(),
            'classifier': self.classifier,
            'class_to_idx': self.model.class_to_idx,
        }

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        torch.save(model_state, os.path.join(save_dir, self.arch + "_retrained.pth"))


def main():
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", action="store")
    parser.add_argument("--category_names", type=str, help="categories name json file", default="cat_to_name.json")
    parser.add_argument("--save_dir", type=str, help="save directory for checkpoint", default="./")
    parser.add_argument("--learning_rate", type=float, help="learning rate for training", default=0.001)
    parser.add_argument("--hidden_units", type=int, nargs='+',help="hidden units for classifier", default=[3136, 784])
    parser.add_argument("--arch", type=str, help="architecture of classifier model", default="vgg16")
    parser.add_argument("--epochs", type=int, help="training epochs", default=1)
    parser.add_argument("--gpu", action="store_true", help="use GPU if available", default=False)

    args = parser.parse_args()

    if not os.path.exists(args.data_directory):
        raise Exception("Unable to locate data dir '{}'".format(args.data_directory))

    if not os.path.exists(args.labels):
        raise Exception("Unable to locate labels file '{}'".format(args.labels))

    logging.info("Received arguments: {}".format(args))
    trainer = Trainer(args.data_directory, args.category_names)

    trainer.train(arch=args.arch, learning_rate=args.learning_rate, num_epochs=args.epochs, use_gpu=args.gpu, hidden_units=args.hidden_units)
    trainer.test_accuracy()
    trainer.save_checkpoint(save_dir=args.save_dir)

if __name__ == "__main__":
    main()