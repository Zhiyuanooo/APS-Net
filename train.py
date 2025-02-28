import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from models import *
from utils import Trainer
import random
import csv

def default_loader(path, is_img):
    if is_img:
        img = Image.open(path).convert('L')
        img = img.resize((512, 512), Image.LANCZOS)
    else:
        img = Image.open(path).convert('1')
        img = img.resize((128, 128), Image.LANCZOS)
    return img

class MyDataset(Dataset):
    def __init__(self, mode, txt, transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.mode = mode
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(data_root + 'img/' + fn, True)
        tar = self.loader(data_root + 'msk/' + fn, False) 
        
        '''
        angle = random.uniform(-20, 20)
        img = img.rotate(angle)
        tar = tar.rotate(angle)   
            
        if random.randint(0, 1) and self.mode == 'train':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            tar = tar.transpose(Image.FLIP_LEFT_RIGHT)
        '''
        img = self.transform(img)
        tar = torch.from_numpy(np.array(tar, np.float32, copy=False))
        tar = torch.unsqueeze(tar, 0)
        return img, tar, label

    def __len__(self):
        return len(self.imgs)


def main(model):
    train_data = MyDataset(
        mode='train',
        txt=label_root + 'train_label.txt',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.
            transforms.Normalize(mean=[img_mean],
                                 std=[img_std])
        ]))
    test_data = MyDataset(
        mode='test',
        txt=label_root + 'test_label.txt',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean],
                                 std=[img_std])
        ]))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    optimizer = optim.SGD(params=model.parameters(),
                           lr=base_lr, momentum=0.9, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    trainer = Trainer(model, optimizer, save_dir=save_root)
    trainer.loop(max_epoch, train_loader, test_loader, scheduler, save_freq=50)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    torch.backends.cudnn.benchmark = True
    # argparse
    batch_size = 256
    data_root = './samples/'
    label_root = './samples/'
    save_root = './temp/ResNet50_SE'
    img_mean = 0.3309
    img_std = 0.1924 #R2=0.03701776
    model = APSnet50()
    base_lr = 0.1
    max_epoch = 300

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    main(model)