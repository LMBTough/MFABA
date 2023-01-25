import pandas as pd
import pickle as pkl
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision import transforms, datasets
import numpy as np
import os
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageNetDataset(Dataset):
    def __init__(self, train, transform) -> None:
        super().__init__()
        if train != False:
            raise NotImplementedError
        self.data = pd.read_csv('data/mini-imagenet/test.csv')
        self.transform = transform
        self.class_name2label = pkl.load(
            open('data/class_name2label.pkl', 'rb'))
        self.classes = list(self.class_name2label.keys())
        if not os.path.exists(f'data/mini-imagenet/test_imgs.pt') or not os.path.exists(f'data/mini-imagenet/test_labels.pt'):
            self.imgs = list()
            self.labels = list()
            for i, row in tqdm(self.data.iterrows(), total=len(self.data)):
                img = pil_loader(
                    "data/mini-imagenet/images/" + row['filename'])
                img = self.transform(img)
                self.imgs.append(img)
                self.labels.append(int(self.class_name2label[row['label']]))
            self.imgs = torch.stack(self.imgs)
            self.labels = torch.LongTensor(self.labels)
            torch.save(
                self.imgs, f'data/mini-imagenet/test_imgs.pt')
            torch.save(
                self.labels, f'data/mini-imagenet/test_labels.pt')
        else:
            self.imgs = torch.load(
                f'data/mini-imagenet/test_imgs.pt')
            self.labels = torch.load(
                f'data/mini-imagenet/test_labels.pt')

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def get_dataset(name, batch_size):
    if name == "cifar10":
        _normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            _normalizer,
        ]))
    elif name == "cifar100":
        _normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        dataset = datasets.CIFAR100("./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            _normalizer,
        ]))
    elif name == "imagenet":
        _normalizer = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        dataset = ImageNetDataset(train=False, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            _normalizer,
        ]))
    else:
        raise NotImplementedError
    data_min = np.min((0 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    data_max = np.max((1 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, data_min, data_max


def load_test_dataset():
    _normalizer = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _normalizer
    ])
    X = list()
    for img in os.listdir("images"):
        img = pil_loader("images/" + img)
        img = transformer(img)
        X.append(img)
    y = torch.LongTensor([111] * len(X))
    X = torch.stack(X)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    data_min = np.min((0 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    data_max = np.max((1 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    return dataloader, data_min, data_max