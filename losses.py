# -*-coding:utf-8-*-

import torch
import torch.nn.functional as F
from math import exp
import torch.nn as nn
import torchvision.models as models
import os
from math import pi


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19_relu())
        self.criterion = torch.nn.MSELoss()
        self.weights = [1.0/64, 1.0/64, 1.0/32, 1.0/32, 1.0/1]
        self.IN = nn.InstanceNorm2d(512, affine=False, track_running_stats=False)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)
    
    def __call__(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        loss  = self.weights[0] * self.criterion(self.IN(x_vgg['relu1_1']), self.IN(y_vgg['relu1_1']))
        loss += self.weights[1] * self.criterion(self.IN(x_vgg['relu2_1']), self.IN(y_vgg['relu2_1']))
        loss += self.weights[2] * self.criterion(self.IN(x_vgg['relu3_1']), self.IN(y_vgg['relu3_1']))
        loss += self.weights[3] * self.criterion(self.IN(x_vgg['relu4_1']), self.IN(y_vgg['relu4_1']))
        loss += self.weights[4] * self.criterion(self.IN(x_vgg['relu5_1']), self.IN(y_vgg['relu5_1']))

        return loss


class VGG19_relu(torch.nn.Module):
    def __init__(self):
        super(VGG19_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg19(pretrained=True)
        # cnn.load_state_dict(torch.load(os.path.join('./models/', 'vgg19-dcbb9e9d.pth')))
        cnn = cnn.to(self.device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class AngularLoss(torch.nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos_criterion = torch.nn.CosineSimilarity(dim=1)
        cos = cos_criterion(feature1, feature2)
        clip_bound = 0.999999
        cos = torch.clamp(cos, -clip_bound, clip_bound)
        if False:
            return 1 - torch.mean(cos)
        else:
            return torch.mean(torch.acos(cos)) * 180 / pi


class MultiscaleRecLoss(nn.Module):
    def __init__(self, scale=3, rec_loss_type='l1', multiscale=True):
        super(MultiscaleRecLoss, self).__init__()
        self.multiscale = multiscale
        if rec_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif rec_loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif rec_loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(rec_loss_type))
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        if self.multiscale:
            self.weights = [1.0, 1.0/2, 1.0/4]
            self.weights = self.weights[:scale]

    def forward(self, input, target):
        loss = 0
        pred = input.clone()
        gt = target.clone()
        if self.multiscale:
            for i in range(len(self.weights)):
                loss += self.weights[i] * self.criterion(pred, gt)
                if i != len(self.weights) - 1:
                    pred = self.downsample(pred)
                    gt = self.downsample(gt)
        else:
            loss = self.criterion(pred, gt)
        return loss


def hingeloss(x, y, mode='fake'):
    if mode == 'fake':
        return torch.mean(nn.ReLU()(x + y))
    elif mode == 'real':
        return torch.mean(nn.ReLU()(x - y))
    else:
        raise NotImplementedError("=== Mode [{}] is not implemented. ===".format(mode))

def diff(x, y, mode=True):
    if mode:
        return x - torch.mean(y)
    else:
        return torch.mean(x) - y

def calc_l2(x, y, mode=False):
    if mode:
        return torch.mean((x - y) ** 2)
    else:
        return torch.mean((x + y) ** 2)


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        elif gan_mode == 'rahinge':
            pass
        elif gan_mode == 'rals':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        if self.gan_mode == 'original': # cross entropy loss
            if for_real:
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(real_preds, target_tensor)
                return loss
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(fake_preds, target_tensor)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'ls':   
            if for_real:  
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                return F.mse_loss(real_preds, target_tensor)
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                return F.mse_loss(fake_preds, target_tensor)
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'hinge':
            if for_real:
                if for_discriminator:                
                    if target_is_real:
                        minval = torch.min(real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(real_preds)
                return loss
            elif for_fake:
                if for_discriminator:                
                    if target_is_real:
                        minval = torch.min(fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(fake_preds)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'rahinge':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))
                return loss / 2
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
                return loss / 2
        elif self.gan_mode == 'rals':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff - 1) ** 2) + torch.mean((f_r_diff + 1) ** 2)
                return loss / 2
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff + 1) ** 2) + torch.mean((f_r_diff - 1) ** 2)
                return loss / 2
        else:
            # wgan
            if for_real:
                if target_is_real:
                    return -real_preds.mean()
                else:
                    return real_preds.mean()
            elif for_fake:
                if target_is_real:
                    return -fake_preds.mean()
                else:
                    return fake_preds.mean()
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")

    def __call__(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        ## computing loss is a bit complicated because |input| may not be
        ## a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(real_preds, list):
            loss = 0
            for (pred_real_i, pred_fake_i) in zip(real_preds, fake_preds):
                if isinstance(pred_real_i, list):
                    pred_real_i = pred_real_i[-1]
                if isinstance(pred_fake_i, list):
                    pred_fake_i = pred_fake_i[-1]

                loss_tensor = self.loss(pred_real_i, pred_fake_i, target_is_real, for_real, for_fake, for_discriminator)

                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss
        else:
            return self.loss(real_preds, target_is_real, for_discriminator)


