# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:23:15 2021
DOG VS CAT (猫狗大战)
Author：Wang Weixing (王卫星)
"""

import csv
from PIL import Image
import os

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
                img = img.resize((128, 128))
                if img.mode != "RGB":
                    img = img.convert('RGB')
                img.save(os.path.join(process_path, str(i) + ".jpg"))
                print(os.path.join(process_path, str(i) + ".jpg"))
                writer.writerow({'category': category, 'category_num':category_num ,'index': i})
                i = i + 1

