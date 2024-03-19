"""
# > Model architecture of Udepth 
#    - Paper: https://arxiv.org/pdf/2209.12358.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
#setting the path to the project root for facillitating importing
script_path = os.path.abspath(__file__)
project_folder = os.path.abspath(os.path.join(os.path.dirname(script_path), '../..'))
# print(project_folder)
sys.path[0] = project_folder

from models.Udepth.miniViT import mViT


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1280, decoder_width = .6):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)
        
        self.up0 = UpSample(skip_input=features//1 + 320, output_features=features//2)
        self.up1 = UpSample(skip_input=features//2 + 160, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 64, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 32, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 +  24, output_features=features//8)
        self.up5 = UpSample(skip_input=features//8 +  16, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4,x_block5,x_block6 = features[2], features[4], features[6], features[9], features[15],features[18],features[19]
        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5)
        x_d2 = self.up1(x_d1, x_block4)
        x_d3 = self.up2(x_d2, x_block3)
        x_d4 = self.up3(x_d3, x_block2)
        x_d5 = self.up4(x_d4, x_block1)
        x_d6 = self.up5(x_d5, x_block0)
        
        return x_d6

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        import torchvision.models as models
        self.original_model = models.mobilenet_v2( pretrained=True )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

class UDepth(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.001, max_val=1, norm='linear'):
        super(UDepth, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder()
        self.adaptive_bins_layer = mViT(48, n_query_channels=48, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=48, norm=norm)
        self.decoder = Decoder()
        self.conv_out = nn.Sequential(nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        unet_out = self.decoder(self.encoder(x))
        # print(unet_out.shape)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)

        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m

class UDepth_SFM(nn.Module):
    def __init__(self, load_pretrained=False):
        super(UDepth_SFM, self).__init__()
        self.model = UDepth.build(100)
        if load_pretrained:
            weights = torch.load('models/Udepth/model_RGB.pth')
            model.load_state_dict(weights)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        bin_edges, pred = self.model(x)
        pred = self.upsample(pred)
        return bin_edges, pred


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # model = UDepth.build(100)
    x = torch.rand(2, 3, 256, 464)
    print(x.shape)
    # bins, pred = model(x)
    # print(bins.shape, pred.shape)

    model = UDepth_SFM(load_pretrained=True)
    # bins, pred = model(x)
    # print(bins.shape, pred.shape)
    # # plt.imshow(pred[0][0].detach().cpu().numpy())
    # plt.show()
    
    from datasets.sequence_folders import SequenceFolder
    import custom_transforms

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])

    transform = custom_transforms.Compose([custom_transforms.ArrayToTensor()])
    dataset = SequenceFolder('data/Eiffel_tower_ready_small_set', transform=transform)
    import torch.utils.data
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        print(tgt_img.shape)
        _, tgt_depth = model(tgt_img) 
        print(tgt_depth.shape)
        break
    # print(np.moveaxis(tgt_img.detach().cpu().numpy(), 1, 3).shape)
    # print(tgt_img.max())
    # plt.imshow(np.moveaxis(tgt_img.detach().cpu().numpy(), 1, 3)[0].astype(np.float32)*255)
    # # plt.show()
    # import matplotlib
    # matplotlib.use('Agg')
    
    plt.imshow(tgt_depth.detach().cpu().snumpy()[0][0])
    plt.imwrite('a.png')
    # plt.show()
    pass
