import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from thop import profile
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


#CBAM改

class Channel(nn.Module ):
    def __init__(self,inchannel,reduction=16):
        super(Channel, self).__init__()
        radio=reduction //2


        self.gap=nn.AdaptiveAvgPool2d(1)
        self.gmp=nn.AdaptiveMaxPool2d (1)

        self.mlp=nn.Sequential (
            nn.Conv2d(inchannel ,inchannel //radio ,1,1,0,bias= False ),
            nn.ReLU (),
            nn.Conv2d (inchannel //radio,inchannel//reduction ,1,1,0,bias= False ),
            nn.ReLU (),
            nn.Conv2d (inchannel //reduction ,inchannel//radio ,1,1,0,bias= False ),
            nn.ReLU (),
            nn.Conv2d (inchannel//radio,inchannel ,1,1,0,bias= False )
        )
        self.sig=nn.Sigmoid ()

    def forward(self,x):

        gap=self.gap (x)
        gmp=self.gmp(x)

        gap=self.mlp(gap)
        gmp=self.mlp(gmp)


        return self.sig(gap+gmp)


class Spatial(nn.Module ):
    def __init__(self):
        super(Spatial, self).__init__()

        self.conv=nn.Sequential (

            nn.Conv2d (in_channels= 2,out_channels= 2,kernel_size= 3,stride= 1,padding= 1,bias= False ),
            nn.ReLU (),
            nn.Conv2d (in_channels= 2,out_channels= 2,kernel_size= 3,stride= 1,padding= 1,bias= False ),
            nn.ReLU (),
            nn.Conv2d (in_channels= 2,out_channels= 1,kernel_size= 3,stride= 1,padding= 1,bias= False )
        )

        self.sig=nn.Sigmoid ()

    def forward(self,x):

        max_pool,_=torch.max (x,dim=1,keepdim= True)
        avg_pool  =torch.mean(x,dim=1,keepdim= True)

        cc=torch.cat([max_pool ,avg_pool ],dim= 1)

        cc=self.conv (cc)

        cc=self.sig(cc)

        return cc


class CBAMs(nn.Module ):
    def __init__(self,inchannel,reduction):
        super(CBAMs, self).__init__()

        self.channel=Channel (inchannel ,reduction )
        self.spatial=Spatial ()

    def forward(self,x):

        ca=self.channel (x)
        out=ca*x
        sa=self.spatial (out)
        out=out*sa

        return out

class Dblock(nn.Module):
    def __init__(self, channel,d_bin):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=d_bin[0], padding=d_bin[0])
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=d_bin[1], padding=d_bin[1])
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=d_bin[2], padding=d_bin[2])
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=d_bin[3], padding=d_bin[3])
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        # self.att=CBAMs(channel,16)

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class RFE_LINKNET(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RFE_LINKNET, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock1=Dblock(64,d_bin=[1,32,32,64])
        # self.dblock2=Dblock(128,d_bin=[1,16,16,32])
        # self.dblock3=Dblock(256,d_bin=[1,4,8,16])
        # self.dblock4=Dblock(512,d_bin=[1,2,4,8])


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.att1=CBAMs(64,16)
        self.att2 = CBAMs(128, 16)
        self.att3 = CBAMs(256, 16)
        self.att4 = CBAMs(512, 16)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center


        # Decoder
        d4 = self.decoder4(self.att4(e4))
        d3 = self.decoder3(self.att3(d4))
        d2 = self.decoder2(self.att2(d3) )
        d1 = self.decoder1(self.att1(d2))

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
if __name__ == '__main__':
    a=torch.randn((1,3,1024,1024))
    model=RFE_LINKNET()
    out=model(a)
    flops, params = profile(model, inputs=(a,))
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("------|-----------|------")
    print("%s | %.7f | %.7f" % ("模型  ", params / (1000 ** 2), flops / (1000 ** 3)))
