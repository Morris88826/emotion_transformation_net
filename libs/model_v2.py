import torch
import torch.nn as nn
import pickle as pkl

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

class EmotionTransformNet(nn.Module):
    def __init__(self, latent_features=128, output_features=244):
        super(EmotionTransformNet, self).__init__()
        self.output_features = output_features
        
        self.layers = nn.Sequential(*[
            nn.Linear(in_features=latent_features, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),  
            nn.Linear(in_features=256, out_features=output_features)
        ])
    def forward(self, images, feat_latent, mean_faces, eigenfaces):

        p_images = torch.bmm(eigenfaces, (images.reshape((-1, 48*48))-mean_faces).unsqueeze(-1))
        p_images = p_images.reshape((-1,self.output_features))
        p_images = p_images.unsqueeze(1)

        weights = self.layers(feat_latent)
        weights = weights.unsqueeze(1)
        # print(t_images.shape)
        # print(weights.shape)
        # raise NotImplementedError
        
        pred = mean_faces + torch.bmm(0.7*(weights)+0.3*p_images, eigenfaces).squeeze()
        pred = pred.reshape((-1, 1, 48, 48))
        return pred

class FeatCaptureNet(nn.Module):
    def __init__(self, latent_features, only_latent=False):
        super(FeatCaptureNet, self).__init__()
        self.encoder = FeatEncoder(latent_features)
        self.decoder = FeatDecoder(latent_features)
        self.only_latent = only_latent
    
    def forward(self, x):
        latent_variable, skip_layers = self.encoder(x)

        if not self.only_latent:
            x_recover = self.decoder(latent_variable, skip_layers)
            return latent_variable, x_recover
        else:
            return latent_variable

class FeatEncoder(nn.Module):
    def __init__(self, latent_features):
        super(FeatEncoder, self).__init__()

        self.latent_features = latent_features
        self.block1 = nn.Sequential(*[
            ConvBlock(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        ])
        self.block2 = nn.Sequential(*[
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        ])

        self.block3 = nn.Sequential(*[
            ConvBlock(in_channels=64, out_channels=128, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        ])

        self.block4 = nn.Sequential(*[
            ConvBlock(in_channels=128, out_channels=512, kernel_size=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        ])

        self.fc_layers = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(512*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_features),
        ])

    def forward(self, x):
        skip1 = x = self.block1(x)
        skip2 = x = self.block2(x)
        skip3 = x = self.block3(x)
        x = self.block4(x)
        x = self.fc_layers(x)
        return x, [skip1, skip2, skip3]


class FeatDecoder(nn.Module):
    def __init__(self, latent_features):
        super(FeatDecoder, self).__init__()

        self.latent_features = latent_features
        self.fc_layers = nn.Sequential(*[
            nn.Linear(self.latent_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512*3*3)
        ])

        self.block4 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.block3 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])

        self.block2 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])

        self.block1 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),  
        ])

        self.conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        
    def forward(self, x, skip_layers):
        x = self.fc_layers(x)
        x = x.reshape((-1, 512, 3, 3))
        x = self.block4(x)
        # x = torch.cat([x, skip_layers[1]], dim=1)
        x = self.block3(x)
        # x = torch.cat([x, skip_layers[0]], dim=1)
        x = self.block2(x)
        # x = torch.cat([x, skip_layers[0]], dim=1)
        x = self.block1(x)
        x = self.conv(x)
        return x




if __name__ == '__main__':

    input = torch.zeros((10, 1, 48, 48))
    mean_face = torch.zeros((10, 1, 48, 48))
    eigenfaces = torch.zeros((10, 227, 48, 48))

    model =  FeatCaptureNet(latent_features=256)

    x = model.forward(input)
