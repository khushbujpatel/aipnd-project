#!/usr/bin/env python3
import json
import logging
import os

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image


class Predictor(object):

    def __init__(self, category_names):
        """
        Sets default device to CPU and initializes category names
        """
        self.device = torch.device("cpu")
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.image_size = 224

        self.load_labels(category_names)

    def load_checkpoint(self, ckpt_path, use_gpu):
        """
        Load trained model's checkpoint
        """
        # Loading weights for CPU model while trained on GPU
        # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
        ckpt_model_state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        self.ckpt_model = models.vgg16(pretrained=True)
        self.ckpt_model.classifier = ckpt_model_state['classifier']
        self.ckpt_model.load_state_dict(ckpt_model_state['state_dict'])
        self.ckpt_model.class_to_idx = ckpt_model_state['class_to_idx']

        return self.ckpt_model

    def load_image(self, image_path):
        """
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        """
        image = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(self.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)])
        image = transform(image)

        return image

    def imshow(self, image, ax=None, title=None):
        """
        Image show for Tensor
        """
        if ax is None:
            _, ax = plt.subplots()

        image = image.numpy().transpose((1, 2, 0))

        mean = np.array(self.mean)
        std = np.array(self.std)

        image = std * image + mean
        image = np.clip(image, 0, 1)

        ax.imshow(image)

        return ax

    def predict(self, image_path, ckpt_path, use_gpu=False, topk=5):
        """
        Predict the class (or classes) of an image using a trained deep learning model.
        """
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info("using device: {}".format(self.device))
        self.load_checkpoint(ckpt_path, use_gpu)

        self.ckpt_model.eval()

        image = self.load_image(image_path)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = self.ckpt_model.forward(image)
            top_prob, top_labels = torch.topk(output, topk)
            top_prob = top_prob.exp()

        class_to_idx_inv = {self.ckpt_model.class_to_idx[k]: k for k in self.ckpt_model.class_to_idx}
        mapped_classes = list()

        for label in top_labels.numpy()[0]:
            if label in class_to_idx_inv:
                mapped_classes.append(class_to_idx_inv[label])

        return top_prob.numpy()[0], mapped_classes

    def load_labels(self, category_names):
        """
        Load category names map {key:value}
        where key is index and value is name of category
        """
        with open(category_names, 'r') as f:
            self.labels = json.load(f)
        return self.labels

def main():
    logging.basicConfig(level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="path to image")
    parser.add_argument("checkpoint", help="trained model")
    parser.add_argument("--top_k", type=int, help="top K results to predict", default=5)
    parser.add_argument("--category_names", type=str, help="categories name json file", default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", help="use GPU if available", default=False)

    args = parser.parse_args()
    logging.info("Args: {}".format(args))

    if not os.path.exists(args.image_path):
        raise Exception("Unable to locate image: {}".format(args.image_path))

    if not os.path.exists(args.checkpoint):
        raise Exception("Unable to locate checkpoint: {}".format(args.checkpoint))

    if not os.path.exists(args.category_names):
        raise Exception("Unable to locate categories name: {}".format(args.category_names))

    predictor = Predictor(args.category_names)

    top_prob, top_classes = predictor.predict(args.image_path, args.checkpoint, args.gpu, args.top_k)

    label = top_classes[0]

    plt.figure(figsize=(6,6))
    sp_img = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
    sp_prd = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

    image = Image.open(args.image_path)
    sp_img.axis('off')
    sp_img.set_title(f'{predictor.labels[label]}')
    sp_img.imshow(image)

    labels = []
    for class_idx in top_classes:
        labels.append(predictor.labels[class_idx])

    yp = np.arange(5)
    sp_prd.set_yticks(yp)
    sp_prd.set_yticklabels(labels)
    sp_prd.set_xlabel('Probability')
    sp_prd.invert_yaxis()
    sp_prd.barh(yp, top_prob, xerr=0, align='center', color='blue')

    plt.show()

if __name__ == "__main__":
    main()
