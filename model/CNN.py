# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:23:15 2021
DOG VS CAT (猫狗大战)
Author：Wang Weixing (王卫星)
"""

import torch
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

batch_size = 128


# 构建数据集
class dataset(Dataset):
    def __init__(self, csv_form, transform=None):
        self.csv = csv_form
        self.transform = transform

    def __getitem__(self, item):
        current_data = self.csv.iloc[item]
        work_dir = "D:\\dataset\\dogVScat\\processed_img\\" + str(current_data['category'])
        file_name = str(current_data['index'])+'.jpg'
        path = os.path.join(work_dir, file_name)
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        label = current_data['category_num']
        return image,label

    def __len__(self):
        return len(self.csv)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=5),   # 15 * 128 *128
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=3)  # 15 * 41 * 41
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(15, 30, kernel_size=5),   # 30 * 37 * 37
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

        self.fc = nn.Sequential(
            nn.Linear(30 * 12 * 12, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

csv = pd.read_csv('D:\dataset\dogVScat\cat_dog.csv')
total_data = dataset(csv, transform=data_tf)

# 80% Training Set，10% Valid Set，10% Test Set
train_size = int(0.8 * len(total_data))
temp_size = len(total_data) - train_size
train_set, temp_set = torch.utils.data.random_split(total_data, [train_size, temp_size],
                                                    generator=torch.Generator().manual_seed(15))
valid_size = int(1 / 2 * temp_size)
test_size = temp_size - valid_size
valid_set, test_set = torch.utils.data.random_split(temp_set, [valid_size, test_size],
                                                    generator=torch.Generator().manual_seed(15))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = Net()

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
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
        if batch_idx % 50 == 49:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 50))
            running_loss = 0.0
    return running_loss


def verification(type):
    correct = 0
    total = 0
    if (type == "test"):
        model_path = '../saved_model/cnn.pkl'
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
    print('Start to train...')
    lost_list = []
    accuracy_list = []
    loops = 25
    for epoch in range(loops):
        print(epoch + 1, '/', loops, '------------')
        lost_list.append(train(epoch) / 50)
        accuracy_list.append(verification("valid"))

    save('../saved_model/cnn.pkl')

    # TEST Function, switch by comment
    # print('Test')
    # verification("test")

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