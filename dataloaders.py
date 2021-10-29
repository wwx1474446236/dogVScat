import torch
import os
import pandas as pd

from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset


batch_size = 512
processed_dir = "D:\\dataset\\dogVScat\\processed_img\\"
csv_path = 'D:\\dataset\\dogVScat\\cat_dog.csv'

# 构建数据集
class dataset(Dataset):
    def __init__(self, csv_form, transform=None):
        self.csv = csv_form
        self.transform = transform

    def __getitem__(self, item):
        current_data = self.csv.iloc[item]
        work_dir = processed_dir + str(current_data['category'])
        file_name = str(current_data['index'])+'.jpg'
        path = os.path.join(work_dir, file_name)
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        label = current_data['category_num']
        return image,label

    def __len__(self):
        return len(self.csv)


data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

csv = pd.read_csv(csv_path)
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