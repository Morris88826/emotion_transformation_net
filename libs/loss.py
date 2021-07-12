import torch
from torch.autograd import Variable
import torchvision.models as models
from collections import namedtuple
import torch
from torchvision import models
dtype = torch.cuda.FloatTensor

def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(logits_real, logits_fake):
    N = logits_fake.shape
    true_labels_ones = Variable(torch.ones(N)).type(dtype)
    true_labels_zeros = Variable(torch.zeros(N)).type(dtype)
    
    loss = bce_loss(logits_real, true_labels_ones) + bce_loss(logits_fake, true_labels_zeros)
    return loss

def generator_loss(logits_fake):
    N = logits_fake.shape
    true_labels = Variable(torch.ones(N)).type(dtype)
    loss = bce_loss(logits_fake, true_labels)
    return loss

class Loss_Functions():
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.style_net = Vgg16(requires_grad=False).to(device)
        self.l2_loss = torch.nn.MSELoss()

    def style_loss(self, image, pred):
        image = torch.cat([image, image, image], dim=1)
        pred = torch.cat([pred, pred, pred], dim=1)
        loss_1 = self.style_net.forward(image)
        loss_2 = self.style_net.forward(pred)

        layer_1 = self.l2_loss(getattr(loss_1, 'relu1_2'), getattr(loss_2, 'relu1_2'))
        layer_2 = self.l2_loss(getattr(loss_1, 'relu2_2'), getattr(loss_2, 'relu2_2'))        
        layer_3 = self.l2_loss(getattr(loss_1, 'relu3_3'), getattr(loss_2, 'relu3_3'))
        layer_4 = self.l2_loss(getattr(loss_1, 'relu4_3'), getattr(loss_2, 'relu4_3'))

        loss = layer_1 + layer_2 + layer_3 + layer_4

        return loss
        

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out