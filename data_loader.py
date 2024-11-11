#-*-coding:utf-8-*-

from pathlib import Path
from itertools import chain
import os

from munch import Munch
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2 = [], []

        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            if idx == 0:
                fnames += cls_fnames
            elif idx == 1:
                fnames2 += cls_fnames

        return list(zip(fnames, fnames2))

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        img_name = os.path.splitext(os.path.basename(fname2))[0]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, img_name

    def __len__(self):
        return len(self.samples)


def get_train_loader(root, img_size=512, resize_size=256, batch_size=8, shuffle=True, num_workers=8, drop_last=True):
    transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.Resize([resize_size, resize_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ReferenceDataset(root, transform)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)


def get_test_loader(root, img_size=512, batch_size=8, shuffle=False, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ReferenceDataset(root, transform)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_refs(self):
        try:
            x, y, name = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, name = next(self.iter)
        return x, y, name

    def __next__(self):
        x, y, img_name = self._fetch_refs()
        x, y = x.to(self.device), y.to(self.device)
        inputs = Munch(img_exp=x, img_raw=y, img_name=img_name)
        return inputs

