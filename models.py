#-*- codign:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools


class Generator(nn.Module):
    """Generator network"""
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn):
        super(Generator, self).__init__()

        ###### encoder
        self.enc1 = ConvBlock(in_channels=3,          out_channels=conv_dim* 1, kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*3 --> 256*256*32
        self.enc2 = ConvBlock(in_channels=conv_dim*1, out_channels=conv_dim* 2, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*32 --> 128*128*64
        self.enc3 = ConvBlock(in_channels=conv_dim*2, out_channels=conv_dim* 4, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*64 --> 64*64*128
        self.enc4 = ConvBlock(in_channels=conv_dim*4, out_channels=conv_dim* 8, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*128 --> 32*32*256
        self.enc5 = ConvBlock(in_channels=conv_dim*8, out_channels=conv_dim*16, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*256 --> 16*16*512

        ###### decoder
        self.upsample1 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim*16, conv_dim*8, 1, 1, 0, 1, True, use_sn))
        self.upsample2 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 8, conv_dim*4, 1, 1, 0, 1, True, use_sn)) 
        self.upsample3 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 4, conv_dim*2, 1, 1, 0, 1, True, use_sn)) 
        self.upsample4 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 2, conv_dim*1, 1, 1, 0, 1, True, use_sn)) 

        self.dec1 = ConvBlock(in_channels=conv_dim*16, out_channels=conv_dim*8, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*512 --> 32*32*256
        self.dec2 = ConvBlock(in_channels=conv_dim* 8, out_channels=conv_dim*4, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*256 --> 64*64*128
        self.dec3 = ConvBlock(in_channels=conv_dim* 4, out_channels=conv_dim*2, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*128 --> 128*128*64
        self.dec4 = ConvBlock(in_channels=conv_dim* 2, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*64 --> 256*256*32
        self.dec5 = nn.Sequential(
            SNConv(in_channels=conv_dim*1, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False), 
            SNConv(in_channels=conv_dim*1, out_channels=3,          kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False), 
            nn.Tanh()
        )

        self.ga5 = GAM(conv_dim*16, conv_dim*16, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga4 = GAM(conv_dim* 8, conv_dim* 8, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga3 = GAM(conv_dim* 4, conv_dim* 4, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga2 = GAM(conv_dim* 2, conv_dim* 2, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga1 = GAM(conv_dim* 1, conv_dim* 1, reduction=8, bias=False, use_sn=use_sn, norm=True)
    
    def forward(self, x):
        ### encoder
        x1 = self.enc1( x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x5 = self.ga5(x5)
        
        ### decoder
        y1 = self.upsample1(x5)
        y1 = torch.cat([y1, self.ga4(x4)], dim=1)
        y1 = self.dec1(y1)

        y2 = self.upsample2(y1)
        y2 = torch.cat([y2, self.ga3(x3)], dim=1)
        y2 = self.dec2(y2)

        y3 = self.upsample3(y2)
        y3 = torch.cat([y3, self.ga2(x2)], dim=1)
        y3 = self.dec3(y3)

        y4 = self.upsample4(y3)
        y4 = torch.cat([y4, self.ga1(x1)], dim=1)
        y4 = self.dec4(y4)

        res = self.dec5(y4.mul(x1))

        out = torch.clamp((res + x), min=-1.0, max=1.0) 

        return out


class SNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn):
        super(SNConv, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
        )
    def forward(self, x):
        return self.main(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun, use_sn):
        super(ConvBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        main = []
        main.append(nn.ReflectionPad2d(self.padding))
        main.append(SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))
        norm_fun = get_norm_fun(norm_fun)
        main.append(norm_fun(out_channels))
        main.append(get_act_fun(act_fun))
        self.main = nn.Sequential(*main)
        
    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(Discriminator, self).__init__()
        
        # scale 1 and prediction of scale 1         128
        d_1 = [dis_conv_block(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 2       64
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=7, stride=2, padding=3, dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]
        
        # scale 3 and prediction of scale 3          32
        d_3 = [dis_conv_block(in_channels=conv_dim* 2, out_channels=conv_dim* 4, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim* 4, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]
        
        # scale 4  16
        d_4 = [dis_conv_block(in_channels=conv_dim* 4, out_channels=conv_dim* 8, kernel_size=5, stride=2, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 5 and prediction of scale 5         8
        d_5 = [dis_conv_block(in_channels=conv_dim* 8, out_channels=conv_dim* 16, kernel_size=5, stride=2, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_5_pred = [dis_pred_conv_block(in_channels=conv_dim* 16, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)
        self.d5 = nn.Sequential(*d_5)
        self.d5_pred = nn.Sequential(*d_5_pred)
    
    def forward(self, x):
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)
        
        ds5 = self.d5(ds4)
        ds5_pred = self.d5_pred(ds5)

        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred, ds5_pred]


def dis_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun, use_sn):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))
    norm_fun = get_norm_fun(norm_fun)
    main.append(norm_fun(out_channels))
    main.append(get_act_fun(act_fun))
    main = nn.Sequential(*main)
    return main


def dis_pred_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, type):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias))
    if type in ['ls', 'rals']:
        main.append(nn.Sigmoid())
    elif type in ['hinge', 'rahinge']:
        main.append(nn.Tanh())
    else:
        raise NotImplementedError("Adversarial loss [{}] is not found".format(type))
    main = nn.Sequential(*main)
    return main
    

def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return out


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.data.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class BasicBlock(nn.Module):
    def __init__(self, channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制
        return self.relu(out + x)


class GAM(nn.Module):
    """Global attention module"""
    def __init__(self, in_nc, out_nc, reduction=8, bias=False, use_sn=False, norm=False):
        super(GAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_nc*2, out_channels=in_nc//reduction, kernel_size=1, stride=1, bias=bias, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_nc//reduction, out_channels=out_nc, kernel_size=1, stride=1, bias=bias, padding=0, dilation=1),
        )
        self.fuse = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=in_nc * 2, out_channels=out_nc, kernel_size=1, stride=1, bias=True, padding=0, dilation=1), use_sn),
        )
        self.in_norm = nn.InstanceNorm2d(out_nc)
        self.norm = norm
        self.CBAM = BasicBlock(out_nc)
    def forward(self, x):
        x_mean, x_std = calc_mean_std(x)
        out = self.conv(torch.cat([x_mean, x_std], dim=1))
        # out = self.conv(x_mean)
        out = self.fuse(torch.cat([x, out.expand_as(x)], dim=1))
        if self.norm:
            out = self.in_norm(out) 
        out = self.CBAM(out)
        return out


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun_type == 'Swish':
            return Swish()
        elif act_fun_type == 'SELU':
            return nn.SELU(inplace=True)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type()


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun
    