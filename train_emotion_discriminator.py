import time
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from libs.model import Discriminator, EmotionDiscriminator
from libs.dataloader import FER2013_loader
import seaborn as sn
import pandas as pd
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
    model = EmotionDiscriminator().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    cross_entropy_loss = nn.CrossEntropyLoss()
    
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
    if not os.path.exists('./checkpoints/emotionDiscriminator'):
        os.mkdir('./checkpoints/emotionDiscriminator')
    save_dir = './checkpoints/emotionDiscriminator/{}'.format(out_name)
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
            pred = model.forward(images)

            # Calculating loss and backpropagation
            loss = cross_entropy_loss(pred, gt_labels)
            loss.backward()
            training_loss += (loss.item())/len(train_loader)
            optimizer.step()

            # Save Tensorboard
            train_step += 1
            writer.add_scalar('Loss/train', loss.item(), train_step)   

            # print('Epoch: {}, Batch: {}/{}, Loss:{:.3f}'.format(e, batch_id, len(train_loader), loss.item()))             
        # print('Overall loss: {}, time elapse: {:.3f}'.format(training_loss, time.time()-start_t))

        # Validation
        model.eval()
        validation_loss = 0.0
        # start_t = time.time()
        num_corrects = 0 
        with torch.no_grad():
            for batch_id, data in enumerate(val_loader):
                # Load input variable
                images, gt_labels = data
                images = torch.unsqueeze(images, 1)
                images = Variable(images).type(torch.float).to(device)
                gt_labels = Variable(gt_labels).type(torch.long).to(device)

                # Model forward pass
                pred = model.forward(images)

                # Accuracy
                pred_labels = torch.argmax(pred, dim=1)
                num_corrects += torch.sum(pred_labels == gt_labels).item()

                # Calculating loss and backpropagation
                loss = cross_entropy_loss(pred, gt_labels)
                validation_loss += (loss.item())/len(val_loader)

                # Save Tensorboard
                val_step += 1
                writer.add_scalar('Loss/Validation', loss.item(), val_step)   
                
                # print('Epoch: {}, Batch: {}/{}, Loss:{:.3f}'.format(e, batch_id, len(val_loader), loss.item()))             
            
        print('Epoch: {}, train_loss: {:.3f}, val_loss: {:.3f}, accuracy: {:.3f}%({}/{}), time elapse: {:.3f}s'.format(e, training_loss, validation_loss, num_corrects*100/val_data.dataset_length, num_corrects, val_data.dataset_length, time.time()-start_t))
        
        # Save model:
        if validation_loss < best_loss:
            best_model['best_e'] = e
            best_loss = validation_loss
        best_model['loss'] = validation_loss
        best_model['model'] = model.state_dict()
        save_name = save_dir + "/epoch_{}.pth".format(e)

        if (e+1)%10 == 0:
            torch.save(best_model, save_name)

def test(config, epoch, test_idx=None, use_best_two=False, save=True):
    # Using device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # Dataset
    test_data = FER2013_loader(dataset_type='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=10)

    # Load Model
    checkpoint_path = './checkpoints/emotionDiscriminator/{}/epoch_{}.pth'.format(config['name'], epoch)
    ckpt = torch.load(checkpoint_path)

    model = EmotionDiscriminator()
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    num_corrects = 0 
    confusion_matrix = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
    }
    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            # Load input variable
            images, gt_labels = data
            images = torch.unsqueeze(images, 1)
            images = Variable(images).type(torch.float).to(device)
            gt_labels = Variable(gt_labels).type(torch.long).to(device)

            # Model forward pass
            pred = model.forward(images)

            # Accuracy
            pred_labels = torch.argmax(pred, dim=1)
            best_three = list(torch.argsort(pred,descending=True)[0,:2].detach().cpu().numpy())
            confusion_matrix[gt_labels.item()].append(pred_labels.item())

            if use_best_two:
                if gt_labels.item() in best_three:
                    num_corrects += 1
            else:
                if gt_labels.item() == best_three[0]:
                    num_corrects += 1

    if save:
        CM = np.zeros((7, 7))
        sorted_keys = np.array(sorted(confusion_matrix.keys()))    
        
        for i in range(sorted_keys.shape[0]):
            key = sorted_keys[i]
            cid = (np.argwhere(sorted_keys==key).item())
            for p in confusion_matrix[key]:
                pid = (np.argwhere(sorted_keys==p).item())
                CM[cid, pid] += 1
        CM = CM/np.sum(CM, axis=1)
        df_cm = pd.DataFrame(CM, index = [test_data.label_dict[i] for i in list(sorted_keys)],
                        columns = [test_data.label_dict[i] for i in list(sorted_keys)])
        plt.figure(figsize = (10,7))
        plt.title('Confusion Matrix, Accuracy:{:.1f}%'.format((num_corrects/len(test_loader))*100))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.savefig('Confusion Matrix.png')

    return (num_corrects/len(test_loader)), num_corrects, len(test_loader)


if __name__ == "__main__":

    config = {
        'epochs': 1000,
        'learning_rate': 5e-5,
        'name': 'emotion_2'
    }
    # train(config)

    accuracy, num_corrects, data_size = test(config, epoch=309, use_best_two=False)
    print("Accuracy: {}%({}/{})".format(accuracy*100, num_corrects, data_size))