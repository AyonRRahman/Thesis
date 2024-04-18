import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import torchvision  

import os 
import sys

try:
    from .miniViT import mViT
    from .DispNetS import *
except:
    from miniViT import mViT
    from DispNetS import *

    

class DispNet_mvit(DispNetS):
    def __init__(self,n_bins=100, min_val=0.001, max_val=1, norm='linear', rmi=False):
        # print('hello')
        
        super(DispNet_mvit, self).__init__(rmi = rmi)
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        
        self.adaptive_bins_layer = mViT(32, n_query_channels=48, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=48, norm=norm)
        self.conv_out = nn.Sequential(nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x):
        if self.convert is not None:
            x = self.convert(x)
        
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(out_iconv2)

        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        pred = self.upsample(pred)
        return [pred]

if __name__=="__main__":
    model = DispNet_mvit(rmi=True)  
    x = torch.rand(2, 3, 256, 464)
    
    print(model(x)[0].shape)