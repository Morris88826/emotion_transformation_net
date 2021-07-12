import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
import torchvision.models as pretrained_models



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batchnorm=True, padding=0):
        super(ConvBlock, self).__init__()

        if batchnorm:
            self.layer = nn.Sequential(*[
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])
        else:
            self.layer = nn.Sequential(*[
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True),
                nn.ReLU()
            ])
    def forward(self, x):
        return self.layer(x)

class FeatEncoder(nn.Module):
    def __init__(self, latent_features):
        super(FeatEncoder, self).__init__()

        self.latent_features = latent_features
        self.num_classes = 7
        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=1, out_channels=64, kernel_size=3),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            ConvBlock(in_channels=256, out_channels=512, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_features),
        ])

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.num_classes = 7
        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=1, out_channels=64, kernel_size=3),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            ConvBlock(in_channels=256, out_channels=512, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1)
        ])
        
    
    def forward(self, x):
        x = self.layers(x)
        return x
    

class EmotionDiscriminator(nn.Module):
    def __init__(self):
        super(EmotionDiscriminator, self).__init__()

        self.num_classes = 7
        
        self.conv_block = nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=9, dilation=2)
        self.backbone = pretrained_models.resnet18(pretrained=True)
        fc_in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_in_feat, self.num_classes)
        self.dropout = nn.Dropout(0.2)       


    def forward(self, x):
        x = self.conv_block(x)
        x = self.dropout(x)
        x = self.backbone(x)
        x = self.dropout(x)
        return x

class FaceDiscriminator(nn.Module):
    def __init__(self):
        super(FaceDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(5184, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class Generator(nn.Module):
    def __init__(self, n_eigenfaces):
        super(Generator, self).__init__()
        self.conv_block_1_1 = ConvBlock(in_channels=(1+n_eigenfaces), out_channels=128, padding=2, kernel_size=5)
        self.conv_block_2_1 = ConvBlock(in_channels=128, out_channels=64, padding=2, kernel_size=5)
        self.conv_block_3_1 = ConvBlock(in_channels=64, out_channels=32, padding=2, kernel_size=5)
        
        self.conv_block_1_2 = ConvBlock(in_channels=1, out_channels=16, padding=2, kernel_size=5)
        self.conv_block_2_2 = ConvBlock(in_channels=16, out_channels=32, padding=2, kernel_size=5)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.conv_T_1 = nn.ConvTranspose2d(in_channels=64, out_channels=16, stride=2, kernel_size=2)
        self.conv_T_2 = nn.ConvTranspose2d(in_channels=(16+16), out_channels=1, stride=2, kernel_size=2)
        self.batchnorm = nn.BatchNorm2d(num_features=16)

    def forward(self, image, mean_face, eigenfaces):
        x = torch.cat([mean_face, eigenfaces], dim=1)
        x = self.conv_block_1_1(x)
        x = self.maxpool(x)
        x = self.conv_block_2_1(x)
        x = self.maxpool(x)
        x = self.conv_block_3_1(x)

        _x = self.conv_block_1_2(image)
        _x = skip_1 = self.maxpool(_x)
        _x = self.conv_block_2_2(_x)
        _x = self.maxpool(_x)

        new_x = torch.cat([_x, x], dim=1)
        new_x = self.conv_T_1(new_x)
        new_x = self.batchnorm(new_x)
        new_x = self.relu(new_x)
        new_x = torch.cat([new_x,skip_1], dim=1)
        new_x = self.conv_T_2(new_x)

        return new_x

if __name__ == '__main__':

    input = torch.zeros((10, 1, 48, 48))
    mean_face = torch.zeros((10, 1, 48, 48))
    eigenfaces = torch.zeros((10, 227, 48, 48))


    model =  FeatEncoder(latent_features=256)
    # G = Generator(n_eigenfaces=eigenfaces.shape[1])
    # FD = FaceDiscriminator()
    # x = G.forward(input, mean_face, eigenfaces)

    x = model.forward(input)
    print(x.shape)
    # x = FD.forward(x)
    # print(x.shape)