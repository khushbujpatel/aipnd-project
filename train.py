import copy
import json
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# check available device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device available: ", device)

# dataset
data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_size = 224

data_transforms = {
    "train": transforms.Compose([transforms.RandomRotation(25),
                                 transforms.RandomResizedCrop(image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)]),
    "valid": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)]),
    "test" : transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)])
}

image_datasets = {
    "train" : torchvision.datasets.ImageFolder(root=train_dir, transform=data_transforms["train"]),
    "valid" : torchvision.datasets.ImageFolder(root=valid_dir, transform=data_transforms["valid"]),
    "test"  : torchvision.datasets.ImageFolder(root=test_dir, transform=data_transforms["test"])
}

dataloaders = {
    "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=32, shuffle=True),
    "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32),
    "test" : torch.utils.data.DataLoader(image_datasets["test"], batch_size=32)
}

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Analysis
print("train images: ", len(image_datasets["train"]))
print("valid images: ", len(image_datasets["valid"]))
print("test images: ", len(image_datasets["test"]))
print("total labels: ", len(cat_to_name))

### Training
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# hyper parameters
num_epochs = 5
learning_rate = 0.001

# train functions
model_fn = models.vgg16(pretrained=True)

# fixed feature extractor
for param in model_fn.parameters():
    param.requires_grad = False

output_size = len(cat_to_name)
input_size = model_fn.classifier[0].in_features
hidden_size = [
    (input_size // 8),
    (input_size // 32)
]

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_size[0])),
    ('relu1', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.15)),
    ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
    ('relu2', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.15)),
    ('output', nn.Linear(hidden_size[1], output_size)),
    ('softmax', nn.LogSoftmax(dim=1))
]))
model_fn.classifier = classifier

model_fn = model_fn.to(device)

criterion_fn = nn.NLLLoss()
optimizer_fn = optim.Adam(model_fn.classifier.parameters(), lr=learning_rate)

# train model
model = train_model(model_fn, criterion_fn, optimizer_fn, num_epochs)

# Disabling gradient calculation
corrects = 0
total = 0
total_images = len(dataloaders["test"].batch_sampler) * dataloaders["test"].batch_size

# Disabling gradient calculation
with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        corrects += (preds == labels).sum().item()
        total += labels.size(0)

print("Accurately classified {:d}%% of {:d} images".format(100 * corrects // total, total_images))

## Save Checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
model_state = {
    'epoch': num_epochs,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer_fn.state_dict(),
    'classifier': classifier,
    'class_to_idx': model.class_to_idx,
}

torch.save(model_state, "./vgg16_classifier.pth")
