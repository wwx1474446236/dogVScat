# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:23:15 2021
DOG VS CAT (猫狗大战)
Author：Wang Weixing (王卫星)
"""

from model.CNN import data_tf, Net
import torch
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def app(path):
    # if sys.argv[1].startswith('-'):
        # path = sys.argv[1][1:]
    img = Image.open(path)
    img = img.resize((128, 128))
    if img.mode != "RGB":
        img = img.convert('RGB')
    img = data_tf(img)
    model_path = './saved_model/cnn.pkl'
    if torch.cuda.is_available():
        model_test = torch.load(model_path)
    else:
        model_test = torch.load(model_path, map_location=lambda storage, loc: storage)
    with torch.no_grad():
        model_test.eval()
        if torch.cuda.is_available():
            img = img.to(device)
        images = img.unsqueeze(dim=0)
        print(images.shape)
        output = model_test(images)
        _, predicted = torch.max(output.data, dim=1)
        print("category_num is :", predicted.item())
        print("this picture is:")
    if predicted.item() == 0:
        return "Cat"
    else:
        return "Dog"


if __name__ == '__main__':
    # TEST Function, switch by comment
    print('Application')

    print(app('D:\\dataset\\dogVScat\\PetImages\\Cat\\0.jpg'))

