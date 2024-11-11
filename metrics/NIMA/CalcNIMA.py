# -*- coding: utf-8 -*-
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
from .mobile_net_v2 import mobile_net_v2
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def get_mean_score(score):
#     buckets = np.arange(1, 11)
#     mu = (buckets * score).sum()
#     return mu

# def get_std_score(scores):
#     si = np.arange(1, 11)
#     mean = get_mean_score(scores)
#     std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
#     return std

class NIMA(nn.Module):
    def __init__(self, pretrained_base_model=False):
        super(NIMA, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
            transforms.Resize(256),        
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    image = Transform(image)   
    image = image[None]
    return image


def calc_nima(img_path, result_save_path, epoch):

    result_save_path = './results/nima_val_results/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    model = NIMA()
    # print(model)
    model.load_state_dict(torch.load('./metrics/NIMA/pretrain-model.pth'))
    print('======================= start to calculate NIMA =======================')
    model.to(device).eval()

    mean, deviation, total_mean, total_std = 0.0, 0.0, 0.0, 0.0
    epoch_result = result_save_path + 'NIMA_epoch_' + str(epoch) + '_' + '_mean_std.csv'
    epochfile = open(epoch_result, 'w')
    epochfile.write('image_name' + ','+ 'mean' +',' + 'std' +'\n')

    total_result = result_save_path + 'NIMA_total_results_' + 'epoch' + '_mean_std.csv'
    totalfile = open(total_result, 'a+')
    # totalfile.write('epoch' + ',' + 'mean' + ',' + 'std' + '\n')

    test_imgs = [f for f in os.listdir(img_path)]
    for i, img in enumerate(test_imgs):
        image = Image.open(os.path.join(img_path, img))
        image = prepare_image(image).to(device)
        with torch.no_grad():
            preds = model(image).data.cpu().numpy()[0]

        for j, e in enumerate(preds, 1):
            mean += j * e
        
        for k, e in enumerate(preds, 1):
            deviation += (e * (k - mean) ** 2)
        std = deviation ** (0.5)
        epochfile.write(img + ',' + str(round(mean, 6)) + ',' + str(round(std, 6)) + '\n')
        total_mean += mean
        total_std += std
        mean, deviation = 0.0, 0.0
        if i % 50 == 0:
            print("=== NIMA is processing {:>3d}-th image ===".format(i))
    print("======================= Complete the NIMA test of {:>3d} images ======================= ".format(i+1))
    total_mean = total_mean / i
    total_std = total_std / i
    epochfile.write('Average' + ',' + str(round(total_mean, 6)) + ',' + str(round(total_std, 6)) + '\n')
    epochfile.close()
    totalfile.write(str(epoch) + ',' + str(round(total_mean, 6)) + ',' + str(round(total_std, 6)) + '\n')
    totalfile.close()
    return total_mean

    
    