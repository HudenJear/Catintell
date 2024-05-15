import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block2(nn.Module):
    r""" Block2. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-5.
    """

    def __init__(self, dim, layer_scale_init_value=1e-5,usb=True):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim//8,bias=usb)
        self.dwconv2 = nn.Conv2d(dim*2, dim, kernel_size=5, padding=2, groups=dim//8,bias=usb)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim,bias=usb)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim,bias=usb)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv1(x)
        x = self.dwconv2(torch.cat([x, input],dim=1))
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class ConvSequence(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, dim, layer_scale_init_value,usb=True):
        super(ConvSequence, self).__init__()
        self.rdb1 = Block2(dim=dim, layer_scale_init_value=layer_scale_init_value,usb=usb)
        self.rdb2 = Block2(dim=dim, layer_scale_init_value=layer_scale_init_value,usb=usb)
        self.rdb3 = Block2(dim=dim, layer_scale_init_value=layer_scale_init_value,usb=usb)
        

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class CatintellConv_D5(nn.Module):
    def __init__(self, dim=32, stage=4,neck_blocks=2,layer_scale_init_value=1e-5,use_bias=False):
        super(CatintellConv_D5, self).__init__()
        self.dim = dim
        self.stage = stage
        self.necks=neck_blocks
        self.lsiv=layer_scale_init_value
        self.use_bias=use_bias

        # Input projection
        self.in_proj = nn.Conv2d(3,self.dim,5,1,2,bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                ConvSequence(dim=dim_stage, layer_scale_init_value=self.lsiv,usb=self.use_bias),
                nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2
        
        # Bottleneck
        self.bottleneck =  nn.ModuleList([])
        for ind in range(self.necks):
            self.bottleneck.append(ConvSequence(dim=dim_stage, layer_scale_init_value=self.lsiv,usb=self.use_bias))
        

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.append(nn.ModuleList([
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(dim_stage,dim_stage//2,3,1,1,bias=False),
            Block2(dim_stage,layer_scale_init_value=self.lsiv,usb=self.use_bias),

        ]))
        for i in range(stage-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(dim_stage,dim_stage//4,3,1,1,bias=False),
                Block2(dim_stage//2,layer_scale_init_value=self.lsiv,usb=self.use_bias),
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim*2, 3, 5, 1, 2, bias=False)

        #### activation function
        self.acti = nn.GELU()
        self.out_acti=nn.Tanh()

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


        # Bottleneck
        for  neck in self.bottleneck:
            fea=neck(fea)

        # Decoder
        for i, (UpSample, conv,W_SAB) in enumerate(self.decoder_layers):          
            fea = UpSample(fea)
            fea = conv(fea)
            fea = W_SAB(torch.cat([fea, fea_encoder[self.stage-1-i]],dim=1))

        # Output projection
        out = self.out_acti(self.out_proj(fea)) + x
        return out
    

class LabelBindBlock(nn.Module):
    def __init__(self, in_dim=32, label_dim=4,out_dim=32,label_expanding=1,layer_scale_init_value=1e-5,use_bias=False):
        super(LabelBindBlock, self).__init__()
        self.in_dim= in_dim
        self.out_dim = out_dim
        self.label_dim = label_dim
        self.label_ratio=label_expanding
        self.lsiv=layer_scale_init_value
        self.usb=use_bias
        # Input projection
        self.in_proj=Block2(dim=in_dim,  layer_scale_init_value=self.lsiv,usb=self.usb)
        
        # LN for labels
        self.LN=LayerNorm(label_dim*label_expanding,data_format="channels_first")

        # Encoder(binder)
        self.expanding_pw=nn.Linear(self.in_dim+self.label_dim*self.label_ratio,2*(self.in_dim+self.label_dim*self.label_ratio))
        self.layers = nn.ModuleList([
            
            nn.Conv2d(2*(self.in_dim+self.label_dim*self.label_ratio), self.in_dim+self.label_dim*self.label_ratio, 3, 1, 1, bias=False),
            nn.Conv2d(self.in_dim+self.label_dim*self.label_ratio, self.out_dim+self.label_dim*self.label_ratio, 3, 1, 1, bias=False),
            nn.Conv2d(self.out_dim+self.label_dim*self.label_ratio, self.out_dim, 3, 1, 1, bias=False)
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
        ba = bb.repeat(1,1,h*w)
        bc = torch.reshape(ba,((labels.size()[0]),labels.size()[1],h,w))
        bc = self.LN(bc.repeat(1,self.label_ratio,1,1).type(torch.float))
        fea = self.in_proj(x)
     
        # print (bc[:,:,0:4,0:4])
        fea=torch.cat([fea,bc],dim=1).permute(0,2,3,1)
        fea=self.expanding_pw(fea).permute(0, 3, 1, 2)

        for layer in self.layers:
            fea = layer(fea)

        return self.acti(fea)

class Catintell_G5(nn.Module):
    def __init__(self, dim=32, stage=4,label_dim=4,label_expanding=1,neck_blocks=2,layer_scale_init_value=1e-5,use_bias=False):
        super(Catintell_G5, self).__init__()
        self.dim = dim
        self.stage = stage
        self.label_dim=label_dim
        self.label_ex=label_expanding
        self.necks=neck_blocks
        self.lsiv=layer_scale_init_value
        self.usb=use_bias
        # Input projection
        self.in_proj = nn.Conv2d(3,self.dim,5,1,2,bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                ConvSequence(dim=dim_stage, layer_scale_init_value=self.lsiv,usb=self.usb),
                nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bind labels into features
        self.LBB=LabelBindBlock(dim_stage,4,dim_stage,label_expanding=label_expanding,layer_scale_init_value=self.lsiv,use_bias=self.usb)
        
        # Bottleneck
        self.bottleneck =  nn.ModuleList([])
        for ind in range(self.necks):
            self.bottleneck.append(ConvSequence(dim=dim_stage, layer_scale_init_value=self.lsiv,usb=self.usb))
        

        # decode

        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.append(nn.ModuleList([
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dim_stage,dim_stage//2,3,1,1,bias=False),
            Block2(dim=dim_stage, layer_scale_init_value=self.lsiv,usb=self.usb),
        ]))
        for i in range(stage-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(dim_stage,dim_stage//4,3,1,1,bias=False),
                Block2(dim=dim_stage//2, layer_scale_init_value=self.lsiv,usb=self.usb),
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim*2, 3, 5, 1, 2, bias=False)

        #### activation function
        self.acti = nn.GELU()
        self.out_acti=nn.Tanh()

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
        for  neck in self.bottleneck:
            fea=neck(fea)

        # Decoder
        for i, (UpSample, conv,W_SAB) in enumerate(self.decoder_layers):          
            fea = UpSample(fea)
            fea = conv(fea)
            fea = W_SAB(torch.cat([fea, fea_encoder[self.stage-1-i]],dim=1))

        # Output projection
        out = self.out_acti(self.out_proj(fea)) + x
        return out