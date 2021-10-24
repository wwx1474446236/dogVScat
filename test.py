# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:23:15 2021
DOG VS CAT (猫狗大战)
Author：Wang Weixing (王卫星)
"""
from model.CNN import Net, test_loader, valid_loader
import torch
from sklearn.metrics import f1_score

model = Net()

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

def verification(type):
    correct = 0
    total = 0
    if (type == "test"):
        model_path = './saved_model/cnn.pkl'
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


if __name__ == '__main__':
    # TEST Function, switch by comment
    print('Test')
    verification("test")


