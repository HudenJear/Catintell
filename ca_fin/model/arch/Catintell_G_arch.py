import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import einsum
from .ConvNeXt_arch import Block
import math
import warnings


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.GELU()

        # initialization
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x
    
class ConvNeXtBlock(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, dim, drop_path,layer_scale_init_value):
        super(ConvNeXtBlock, self).__init__()
        self.rdb1 = Block(dim=dim, drop_path=drop_path,layer_scale_init_value=layer_scale_init_value)
        self.rdb2 = Block(dim=dim, drop_path=drop_path,layer_scale_init_value=layer_scale_init_value)
        self.rdb3 = Block(dim=dim, drop_path=drop_path,layer_scale_init_value=layer_scale_init_value)
        

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class CatintellConv_G(nn.Module):
    def __init__(self, dim=32, stage=4):
        super(CatintellConv_G, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.in_proj = nn.Conv2d(3,self.dim,3,1,1,bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                ConvNeXtBlock(dim=dim_stage, drop_path=0,layer_scale_init_value=1e-6),
                nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bind labels into features
        self.LBB=LabelBindBlock(dim_stage,4,dim_stage)
        
        # Bottleneck
        self.bottleneck = RRDB(num_feat=dim_stage, num_grow_ch=dim)

        # decode

        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.append(nn.ModuleList([
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dim_stage,dim_stage//2,3,1,1,bias=False),
            nn.GELU(),
            ConvNeXtBlock(dim=dim_stage, drop_path=0,layer_scale_init_value=1e-6),
        ]))
        for i in range(stage-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(dim_stage,dim_stage//4,3,1,1,bias=False),
                nn.GELU(),
                ConvNeXtBlock(dim=dim_stage//2, drop_path=0,layer_scale_init_value=1e-6),
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim*2, 3, 3, 1, 1, bias=False)

        #### activation function
        self.acti = nn.GELU()

    def forward(self, x,labels):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.acti(self.in_proj(x))

        # Encoder
        fea_encoder = []  # [c 2c 4c 8c]
        for (W_SAB, DownSample) in self.encoder_layers:
            fea = W_SAB(fea)
            fea_encoder.append(fea)
            fea = DownSample(fea)

        # label binding
        fea = self.LBB(fea,labels)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (UpSample, conv,gelu,W_SAB) in enumerate(self.decoder_layers):          
            fea = UpSample(fea)
            fea = conv(fea)
            fea = gelu(fea)
            fea = W_SAB(torch.cat([fea, fea_encoder[self.stage-1-i]],dim=1))

        # Output projection
        out = self.out_proj(fea) + x
        return out


class LabelBindBlock(nn.Module):
    def __init__(self, in_dim=32, label_dim=4,out_dim=32,label_expanding=1):
        super(LabelBindBlock, self).__init__()
        self.in_dim= in_dim
        self.out_dim = out_dim
        self.label_dim = label_dim
        # Input projection
        self.in_proj=Block(dim=in_dim, drop_path=0,layer_scale_init_value=1e-6)
        self.label_ratio=label_expanding


        # Encoder(binder)
        self.layers = nn.ModuleList([
            nn.Conv2d(self.in_dim+self.label_dim, self.in_dim+self.label_dim, 3, 1, 1, bias=False),
            nn.Conv2d(self.in_dim+self.label_dim, self.out_dim+self.label_dim, 3, 1, 1, bias=False),
            nn.Conv2d(self.out_dim+self.label_dim, self.out_dim, 3, 1, 1, bias=False)
        ])
       

        #### activation function
        self.acti = nn.GELU()

    def forward(self, x,labels):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b,c,h,w=x.size()
        # print (labels.size())
        bb=labels.unsqueeze(2)
        # print (bb.size())
        ba=bb.repeat(1,1,h*w)
        bc=torch.reshape(ba,((labels.size()[0]),labels.size()[1],h,w))
        fea = self.acti(self.in_proj(x))
        # print (bc[:,:,0:4,0:4])
        fea=torch.cat([fea,bc],dim=1)
        for layer in self.layers:
            fea = layer(fea)

        return self.acti(fea)

class CatintellConv_D(nn.Module):
    def __init__(self, dim=32, stage=4, blocks=3):
        super(CatintellConv_D, self).__init__()
        self.dim = dim
        self.stage = stage
        self.blocks=blocks

        # Input projection
        self.in_proj = nn.Conv2d(3,self.dim,3,1,1,bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                ConvNeXtBlock(dim=dim_stage, drop_path=0,layer_scale_init_value=1e-6),
                nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bind labels into features
        # self.LBB=LabelBindBlock(dim_stage,4,dim_stage)
        
        # Bottleneck
        self.bottleneck = ConvNeXtBlock(dim=dim_stage, drop_path=0,layer_scale_init_value=1e-6)

        # decode

        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.append(nn.ModuleList([
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dim_stage,dim_stage//2,3,1,1,bias=False),
            nn.GELU(),
            ConvNeXtBlock(dim=dim_stage, drop_path=0,layer_scale_init_value=1e-6),
        ]))
        for i in range(stage-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(dim_stage,dim_stage//4,3,1,1,bias=False),
                nn.GELU(),
                ConvNeXtBlock(dim=dim_stage//2, drop_path=0,layer_scale_init_value=1e-6),
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim*2, 3, 3, 1, 1, bias=False)

        #### activation function
        self.acti = nn.GELU()

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.acti(self.in_proj(x))

        # Encoder
        fea_encoder = []  # [c 2c 4c 8c]
        for (W_SAB, DownSample) in self.encoder_layers:
            fea = W_SAB(fea)
            fea_encoder.append(fea)
            fea = DownSample(fea)

        # label binding
        # fea = self.LBB(fea,labels)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (UpSample, conv,gelu,W_SAB) in enumerate(self.decoder_layers):          
            fea = UpSample(fea)
            fea = conv(fea)
            fea = gelu(fea)
            fea = W_SAB(torch.cat([fea, fea_encoder[self.stage-1-i]],dim=1))

        # Output projection
        out = self.out_proj(fea) + x
        return out

