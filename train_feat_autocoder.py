import time
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from libs.model_v2 import FeatCaptureNet
from libs.dataloader import FER2013_loader
import matplotlib.pyplot as plt


train_data = FER2013_loader(dataset_type='train')
train_loader = DataLoader(train_data, batch_size=64)

def train(config):
    # Using device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # Training configuration
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    out_name = config['name']

    # Dataset
    batch_size = 64
    train_data = FER2013_loader(dataset_type='train')
    val_data = FER2013_loader(dataset_type='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=10)

    # Model, Optimizer and loss function
    model = FeatCaptureNet(latent_features=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    # Checkpoints
    best_loss = 1e6
    best_model = {
        'best_e': -1,
        'learning_rate': learning_rate,
        'loss': 1e5,
        'model': None
    }

    # Saved Checkpoint Directory
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./checkpoints/featCaptureNet'):
        os.mkdir('./checkpoints/featCaptureNet')
    save_dir = './checkpoints/featCaptureNet/{}'.format(out_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Tensorboard

    if os.path.exists("./runs/{}".format(out_name)):
        shutil.rmtree("./runs/{}".format(out_name))
    train_step = 0
    val_step = 0
    writer = SummaryWriter("./runs/{}".format(out_name))

    # Training loop
    for e in range(epochs):

        # Train
        model.train()
        training_loss = 0.0
        start_t = time.time()
        for batch_id, data in enumerate(train_loader):
            optimizer.zero_grad()

            # Load input variable
            images, gt_labels = data
            images = torch.unsqueeze(images, 1)
            if gt_labels.shape[0] == 1:
                break
            images = Variable(images).type(torch.float).to(device)
            gt_labels = Variable(gt_labels).type(torch.long).to(device)

            # Model forward pass
            _, pred = model.forward(images)

            pred = pred.squeeze()
            images = images.squeeze()

            # Calculating loss and backpropagation
            loss = l1_loss(pred, images) + l2_loss(pred, images)
            loss.backward()
            training_loss += (loss.item())/len(train_loader)
            optimizer.step()

            # Save Tensorboard
            train_step += 1
            writer.add_scalar('Loss/train', loss.item(), train_step)   

        # Validation
        model.eval()
        validation_loss = 0.0
        # start_t = time.time()
        with torch.no_grad():
            for batch_id, data in enumerate(val_loader):
                # Load input variable
                images, gt_labels = data
                images = torch.unsqueeze(images, 1)
                images = Variable(images).type(torch.float).to(device)
                gt_labels = Variable(gt_labels).type(torch.long).to(device)

                # Model forward pass
                _, pred = model.forward(images)

                pred = pred.squeeze()
                images = images.squeeze()

                # Calculating loss and backpropagation
                loss = l1_loss(pred, images) + l2_loss(pred, images)
                validation_loss += (loss.item())/len(val_loader)

                # Save Tensorboard
                val_step += 1
                writer.add_scalar('Loss/Validation', loss.item(), val_step)   
                
        print('Epoch: {}, train_loss: {:.3f}, val_loss: {:.3f}, time elapse: {:.3f}s'.format(e, training_loss, validation_loss, time.time()-start_t))
        
        # Save model:
        if validation_loss < best_loss:
            best_model['best_e'] = e
            best_loss = validation_loss
        best_model['loss'] = validation_loss
        best_model['model'] = model.state_dict()
        save_name = save_dir + "/epoch_{}.pth".format(e)

        if (e+1)%10 == 0:
            torch.save(best_model, save_name)

def test(test_idx, config, epoch, save=True):

    # Using device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # Dataset
    test_data = FER2013_loader(dataset_type='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=10)

    # Load Model
    checkpoint_path = './checkpoints/featCaptureNet/{}/epoch_{}.pth'.format(config['name'], epoch)
    ckpt = torch.load(checkpoint_path)

    model = FeatCaptureNet(latent_features=128)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # Loss 
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    list_images = []
    list_labels = []
    list_pred_images = []
    for i, data in enumerate(test_loader):
        if i==4:
            break
        # Load input variable
        images, gt_labels = data
        images = torch.unsqueeze(images, 1)
        images = Variable(images).type(torch.float).to(device)
        gt_labels = Variable(gt_labels).type(torch.long).to(device)

        # Model forward pass
        _, pred = model.forward(images)

        pred = pred.squeeze()
        images = images.squeeze()

        loss = l1_loss(pred, images) + l2_loss(pred, images)
        loss = loss.item()
        pred = pred.detach().cpu().numpy()
        images = images.detach().cpu().numpy()

        list_images.append(images)
        list_labels.append(gt_labels.item())
        list_pred_images.append(pred)

    if save:
        fig, axes = plt.subplots(nrows=2, ncols=4)
        for i in range(4):
            image = list_images[i]
            label = list_labels[i]
            pred = list_pred_images[i]

            axes[0, i].imshow(image, cmap='gray')
            axes[0, i].set(title='{}'.format(test_data.label_dict[label]))

            axes[1, i].imshow(pred, cmap='gray')
            axes[1, i].set(title='{}'.format(test_data.label_dict[label]))

        fig.savefig('feat_reconstruct.png')

    return images, pred, loss


if __name__ == "__main__":

    config = {
        'epochs': 200,
        'learning_rate': 5e-5,
        'name': 'no_skip_2'
    }

    # train(config)
    images, pred, loss = test(test_idx=0, config=config, epoch=199)
