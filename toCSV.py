# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:23:15 2021
DOG VS CAT (猫狗大战)
Author：Wang Weixing (王卫星)
"""

import csv
from PIL import Image
import os


img_size_processed = [224, 224]

def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', target_size, (0, 0, 0))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    return new_image

work_dir = "./PetImages"
process_dir = "./processed_img"
with open('cat_dog.csv', 'w', newline='') as csvfile:
    fieldnames = ['category', 'category_num', 'index']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for category in ("Cat", "Dog"):
        path = os.path.join(work_dir, category)
        process_path = os.path.join(process_dir, category)
        i = 0
        if category == 'Cat':
            category_num = 0
        else:
            category_num = 1
        for name in os.listdir(path):
            if name.split('.')[1] == 'jpg':
                img = Image.open(os.path.join(path, name))
                img = pad_image(img, img_size_processed)
                if img.mode != "RGB":
                    img = img.convert('RGB')
                img.save(os.path.join(process_path, str(i) + ".jpg"))
                print(os.path.join(process_path, str(i) + ".jpg"))
                writer.writerow({'category': category, 'category_num':category_num ,'index': i})
                i = i + 1

