# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:23:15 2021
DOG VS CAT (猫狗大战)
Author：Wang Weixing (王卫星)
"""
from dataloaders import test_loader, valid_loader, train_loader
from model.torch_model import initialize_model
import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from sklearn.metrics import f1_score

model_name = 'resnet'  # torchvision has ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
feature_extract = False
model,input_size = initialize_model(model_name, 2, feature_extract, use_pretrained=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # Obtain data and labels for a batch
        inputs, target = data
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            target = target.to(device)
        # Obtain the model prediction results(128, 15)

        outputs = model(inputs)
        # Cross entropy cost function outputs(128,15),target（128）
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0
    return running_loss

def verification(type):
    correct = 0
    total = 0
    if (type == "test"):
        model_path = './saved_model/resnet.pth'
        if torch.cuda.is_available():
            model_test = torch.load(model_path)
        else:
            model_test = torch.load(model_path, map_location=lambda storage, loc: storage)
        loader = test_loader
    else:
        loader = valid_loader
    with torch.no_grad():
        for data in loader:
            if (type == "test"):
                model_test.eval()
            else:
                model.eval()
            images, labels = data
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)
            if (type == "test"):
                outputs = model_test(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 Column is the 0th dimension, row is the first dimension
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Comparison between tensors
        print('accuracy on', type, 'set: %f %% ' % (100 * correct / total))
        if torch.cuda.is_available():
            print('micro: ',
                  f1_score(labels.cuda().data.cpu().numpy(), predicted.cuda().data.cpu().numpy(), average='micro'))
            print('macro: ',
                  f1_score(labels.cuda().data.cpu().numpy(), predicted.cuda().data.cpu().numpy(), average='macro'))
        else:
            print('micro: ', f1_score(labels, predicted, average='micro'))
            print('macro: ', f1_score(labels, predicted, average='macro'))
        return 100 * correct / total



def save(name):
    torch.save(model, name)

if __name__ == '__main__':
    # TRAIN Function, switch by comment
    print('Start to train all layers...')

    model=torch.load('./saved_model/resnet.pth')
    lost_list = []
    accuracy_list = []
    loops = 9
    for epoch in range(loops):
        print(epoch + 1, '/', loops, '------------')
        lost_list.append(train(epoch) / 10)
        accuracy_list.append(verification("valid"))

    save('./saved_model/all_resnet.pth')

    plt.figure(1)
    x = np.arange(1, loops + 1)
    # plt.plot(x, test_list, label="test")
    plt.title("Accuracy of prediction after each training epoch")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.plot(x, accuracy_list)
    # plt.legend(loc="upper left", fontsize=14)
    plt.show()

    plt.figure(2)
    plt.title("Loss after each training epoch")
    plt.plot(x, lost_list, linewidth=2, label="Loss")
    plt.show()