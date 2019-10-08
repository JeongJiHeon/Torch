import torch
import torch.nn as nn



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Encoder_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.Encoder_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.Encoder_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.Encoder_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.Encoder_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.Encoder_layer6 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.Encoder_layer7 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.Encoder_layer8 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        
        
        self.Decoder_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(inplace=True),
            nn.ReLU(True)
        )
        self.Decoder_layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(inplace=True),
            nn.ReLU(True)

        )
        self.Decoder_layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(inplace=True),
            nn.ReLU(True)
        )
        self.Decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(inplace=True),
            nn.ReLU(True)
        )
        self.Decoder_layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.Dropout2d(inplace=True),
            nn.ReLU(True)
        )
        self.Decoder_layer6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 512, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(inplace=True),
            nn.ReLU(True)
        )
        self.Decoder_layer7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.Dropout2d(inplace=True),
            nn.ReLU(True)
        )
        self.Decoder_layer8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 3, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(3),
            nn.Dropout2d(inplace=True),
            nn.Tanh()
        )
        
    def forward(self, inputs):
        e1 = self.Encoder_layer1(inputs)
        e2 = self.Encoder_layer2(e1)
        e3 = self.Encoder_layer3(e2)
        e4 = self.Encoder_layer4(e3)
        e5 = self.Encoder_layer5(e4)
        e6 = self.Encoder_layer6(e5)
        e7 = self.Encoder_layer7(e6)
        e8 = self.Encoder_layer8(e7)

        d1 = self.Decoder_layer1(e8)
        c1 = torch.cat([d1, e7], 1)
        d2 = self.Decoder_layer2(c1)
        c2 = torch.cat([d2, e6], 1)
        d3 = self.Decoder_layer3(c2)
        c3 = torch.cat([d3, e5], 1)
        d4 = self.Decoder_layer4(c3)
        c4 = torch.cat([d4, e4], 1)
        d5 = self.Decoder_layer5(c4)
        c5 = torch.cat([d5, e3], 1)
        d6 = self.Decoder_layer6(c5)
        c6 = torch.cat([d6, e2], 1)
        d7 = self.Decoder_layer7(c6)
        c7 = torch.cat([d7, e1], 1)
        d8 = self.Decoder_layer8(c7)

        return d8
            
            
            
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            

            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2 , padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2 , padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 2 , padding = 1, bias = False),
            nn.Sigmoid()
        )
            
    def forward(self, inputs, label):
        torch.cat([inputs, label], 1).shape
        return self.D(torch.cat([inputs, label], 1))
            
            
            
            
            
            
        
        