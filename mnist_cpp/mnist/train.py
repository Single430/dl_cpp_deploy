#! /bin/python
# -*- coding: utf-8 -*-

"""
  * @author:zbl
  * @file: train.py
  * @time: 2020/07/30
  * @func:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import torch
import numpy as np
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.functional import F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Cnn(nn.Module):

    def __init__(self, labels):
        super(Cnn, self).__init__()
        self.labels = labels
        self.num_label = len(labels)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_label)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    if torch.cuda.is_available():
        images = images.cuda()
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def eval(model):
    model.eval()
    all_loss = 0.0
    acc, count = 0, 0
    for i, data in enumerate(testloader):
        inputs, targets = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            loss = critertion(outputs, targets)
            all_loss += loss
            acc += (outputs.argmax(1) == targets).sum().item()
        count += len(targets)
    return all_loss / len(testset), acc / count


def train():
    running_loss = 0.0
    best_acc = 0.0
    for epoch in range(5):
        for i, data in enumerate(trainloader):
            inputs, targets = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = critertion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss

            if i % 100 == 99:
                writer.add_scalar('training_loss', running_loss / 100, epoch * len(trainloader) + i)
                eval_loss, eval_acc = eval(net)
                writer.add_scalar('eval_loss', eval_loss, epoch * len(trainloader) + i)
                writer.add_scalar('eval_acc', eval_acc, epoch * len(trainloader) + i)

                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(net, inputs.cpu(), targets.cpu()),
                                  global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
                if eval_acc > best_acc and eval_acc > 0.8:
                    best_acc = eval_acc
                    print(f"best_acc: {best_acc:.2f}, eval_loss: {eval_loss}")
                    torch.save(net.state_dict(), f'./model/best_model_{best_acc:.2f}.pth')


def inference():
    state_dict = torch.load("model/best_model_0.99.pth", map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    print(eval(net))

    # TODO: convert
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v
    # # # load params

    # convert pth-model to pt-model
    example = torch.rand(1, 1, 28, 28)
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save("model/mnist.pt")


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
           '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

net = Cnn(classes)
if torch.cuda.is_available():
    net = net.cuda()
critertion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
writer = SummaryWriter('runs/log.log')

print(net, len(trainloader))

train()
print('Finished Training')
inference()
# save img
# from PIL import Image
# for item in testloader:
#     img_list = item[0][0][0].numpy()
#     img_list[img_list > 0] = 0
#     img_list[img_list < 0] = 255
#     img = Image.fromarray(img_list).convert("L")
#     # img = Image.new('L', (28, 28))
#     # img.putpixel()
#     img.save("model/test.png")
#     break
