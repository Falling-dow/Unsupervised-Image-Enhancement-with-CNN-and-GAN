# -*- coding: utf-8 -*-
import argparse
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
from mobile_net_v2 import mobile_net_v2
import numpy as np
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, help='name of method')
parser.add_argument('--dataset', type=str, default='fivek', help='dataset')
parser.add_argument('--test_images', type=str, default='./images/', help='path to folder containing images')
args = parser.parse_args()


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


def main(model):

    mean, deviation = 0.0, 0.0

    result_save_path = './nima_results'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    txt_name = './nima_results/result_' + args.dataset + '_' + args.method + '_mean_std.csv'
    outfile = open(txt_name, 'w')
    outfile.write('image_name' + ','+ 'mean' +',' + 'std' +'\n')

    test_imgs = [f for f in os.listdir(args.test_images)]
    for i, img in enumerate(test_imgs):
        image = Image.open(os.path.join(args.test_images, img))
        image = prepare_image(image).to(device)
        with torch.no_grad():
            preds = model(image).data.cpu().numpy()[0]
        # mean = get_mean_score(preds)
        # std = get_std_score(preds)
        # print('preds: ', preds)

        for j, e in enumerate(preds, 1):
            mean += j * e
            # print('mean: ', round(mean.item(),5))
        
        for k, e in enumerate(preds, 1):
            deviation += (e * (k - mean) ** 2)
            # print('deviation: ', round(deviation.item(),5))
        std = deviation ** (0.5)
        
        outfile.write(img + ',' + str(round(mean, 5)) + ',' + str(round(std, 5)) + '\n')
        print('processing {:>4d}-th image{:s}: mean: {:>2.6f} and std: {:>2.6f}'.format(i+1, img, round(mean,5).item(), round(std, 5).item()))

        mean, deviation = 0.0, 0.0

    outfile.close()


if __name__ == '__main__':
    model = NIMA()
    # print(model)
    model.load_state_dict(torch.load('pretrain-model.pth'))
    print('Successfully loaded pretrained model...')
    model.to(device).eval()

    main(model)
    
    