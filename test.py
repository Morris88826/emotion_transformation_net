import time
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from libs.model import Discriminator, FaceDiscriminator, Generator
from libs.dataloader import FER2013_EmotionTransform
from libs.loss import generator_loss, discriminator_loss
import matplotlib.pyplot as plt

def test(idx, ckpt_path):
    # Using device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # Dataset
    test_data = FER2013_EmotionTransform(dataset_type='test')
    test_loader = DataLoader(test_data, batch_size=1)

    # Model, Optimizer and loss function
    emotion_discriminator = Discriminator()
    pretrain_D_ckpt_path = './checkpoints/discriminator/train_1/epoch_99.pth' 
    emotion_discriminator.load_state_dict(torch.load(pretrain_D_ckpt_path)['model'])
    emotion_discriminator.to(device)

    face_discriminator = FaceDiscriminator()
    face_discriminator.load_state_dict(torch.load(ckpt_path)['FD_state_dict'])
    face_discriminator.to(device)
    
    n_eigenfaces = 244
    generator = Generator(n_eigenfaces)
    generator.load_state_dict(torch.load(ckpt_path)['G_state_dict'])
    generator.to(device)

    for batch_id, data in enumerate(test_loader):
        if batch_id == idx:
            # Load input variable
            images, emotion_id, original_id, mean_face, eigenfaces = data
            images = torch.unsqueeze(images, 1)
            mean_face = torch.unsqueeze(mean_face, 1)

            images = Variable(images).type(torch.float).to(device)
            emotion_id = Variable(emotion_id).type(torch.long).to(device)
            mean_face = Variable(mean_face).type(torch.float).to(device)
            eigenfaces = Variable(eigenfaces).type(torch.float).to(device)

            # Generator
            transformed_images = generator.forward(images, mean_face, eigenfaces)
            return transformed_images, emotion_id, images, original_id


if __name__ == "__main__":
    ckpt_path = './checkpoints/ETNet/train_only_happy/epoch_39.pth'
    transformed_images, emotion_id, images, original_id = test(5, ckpt_path)

    transformed_images = transformed_images.detach().cpu().numpy().squeeze()
    emotion_id = emotion_id.detach().cpu().numpy().squeeze()
    original_id = original_id.detach().cpu().numpy().squeeze()
    images = images.detach().cpu().numpy().squeeze()


    label_dict = {
        0:'Angry',
        1:'Disgust',
        2:'Fear',
        3:'Happy',
        4:'Sad',
        5:'Surprise',
        6:'Neutral',
    }

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(images, cmap='gray')
    axes[0].set(title='Original Image: {}'.format(label_dict[original_id.item()]))
    axes[1].imshow(transformed_images, cmap='gray')
    axes[1].set(title='Emotion transformed: {}'.format(label_dict[emotion_id.item()]))
    fig.savefig('result.png')