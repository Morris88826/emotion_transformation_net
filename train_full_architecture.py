import time
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from libs.model import FaceDiscriminator, EmotionDiscriminator
from libs.model_v2 import  FeatCaptureNet, EmotionTransformNet
from libs.dataloader import FER2013_EmotionTransform
from libs.loss import generator_loss, discriminator_loss, Loss_Functions
import matplotlib.pyplot as plt

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)

def train(config):
    # Using device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # Load dataset
    batch_size = 64
    train_0_data = FER2013_EmotionTransform(dataset_type='train', training_type=0)
    train_0_loader = DataLoader(train_0_data, batch_size=batch_size, shuffle=True, num_workers=10)
    train_1_data = FER2013_EmotionTransform(dataset_type='train', training_type=1)
    train_1_loader = DataLoader(train_1_data, batch_size=batch_size, shuffle=True, num_workers=10)
    train_2_data = FER2013_EmotionTransform(dataset_type='train', training_type=2) # train on random emotion
    train_2_loader = DataLoader(train_2_data, batch_size=batch_size, shuffle=True, num_workers=10)


    # Training configuration
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    out_name = config['name']

    # Model, Optimizer and loss function
    emotion_discriminator = EmotionDiscriminator()
    pretrain_D_ckpt_path = './checkpoints/emotionDiscriminator/emotion_2/epoch_309.pth' 
    emotion_discriminator.load_state_dict(torch.load(pretrain_D_ckpt_path)['model'])
    emotion_discriminator.to(device)
    emotion_discriminator.eval()

    latent_features = 128
    featCaptureNet = FeatCaptureNet(latent_features=latent_features, only_latent=True)
    pretrain_faceCapture_ckpt_path = './checkpoints/featCaptureNet/no_skip_2/epoch_199.pth'
    featCaptureNet.load_state_dict(torch.load(pretrain_faceCapture_ckpt_path)['model'])
    featCaptureNet.to(device)
    featCaptureNet.eval()
    
    n_eigenfaces = 244
    face_discriminator = FaceDiscriminator()
    face_discriminator.apply(weights_init)
    face_discriminator.to(device)
    FD_optim = torch.optim.RMSprop(face_discriminator.parameters(), lr=learning_rate)

    generator = EmotionTransformNet(latent_features=latent_features, output_features=n_eigenfaces)
    generator.apply(weights_init)
    generator.to(device)
    G_optim = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
    cross_entropy_loss = nn.CrossEntropyLoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    loss_func = Loss_Functions()
    
    # Checkpoints
    best_model = {
        'G_state_dict': None,
        'FD_state_dict': None
    }

    # Saved Checkpoint Directory
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./checkpoints/EmotionTransformNet'):
        os.mkdir('./checkpoints/EmotionTransformNet')
    save_dir = './checkpoints/EmotionTransformNet/{}'.format(out_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for e in range(epochs):

        # train on original expression (type 0)
        start_t = time.time()
        training_loss = 0.0
        for batch_id, data in enumerate(train_0_loader):
            # Load input variable
            images, trans_emotion_id, _, mean_face, eigenfaces = data
            images = torch.unsqueeze(images, 1)

            if trans_emotion_id.shape[0] == 1:
                break

            images = Variable(images).type(torch.float).to(device)
            trans_emotion_id = Variable(trans_emotion_id).type(torch.long).to(device)
            mean_face = Variable(mean_face).type(torch.float).to(device)
            eigenfaces = Variable(eigenfaces).type(torch.float).to(device)

            # Face Discriminator
            FD_optim.zero_grad()
            logits_real = face_discriminator.forward(images)
            feat_latent = featCaptureNet.forward(images)
            fake_images = generator.forward(images, feat_latent, mean_face, eigenfaces).detach()
            logits_fake = face_discriminator.forward(fake_images)
            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()        
            FD_optim.step()

            # Generator
            G_optim.zero_grad()
            feat_latent = featCaptureNet.forward(images)
            pred = generator.forward(images, feat_latent, mean_face, eigenfaces)
            gen_logits_fake = face_discriminator(pred)
            pred_labels = emotion_discriminator.forward(pred)

            _style_loss = loss_func.style_loss(images, pred)
            pred = pred.squeeze()
            images = images.squeeze()
            _l1_loss = l1_loss(pred,images)
            _l2_loss = l2_loss(pred,images)
            _emotion_loss = cross_entropy_loss(pred_labels, trans_emotion_id) 
            _face_loss = generator_loss(gen_logits_fake)

            loss = (_l1_loss + _l2_loss) + _emotion_loss + _style_loss + _face_loss 
            loss.backward()
            training_loss += (loss.item()/len(train_0_loader))
            G_optim.step()
        print('Epoch {}, training_type0_loss {:.3f}, time-elapse {:.3f}s'.format(e, training_loss, time.time()-start_t))

        # train on mean face expression (type 1)
        start_t = time.time()
        training_loss = 0.0
        for _ in range(7):
            for batch_id, data in enumerate(train_1_loader):
                # Load input variable
                images, trans_emotion_id, _, mean_face, eigenfaces = data
                images = torch.unsqueeze(images, 1)

                if trans_emotion_id.shape[0] == 1:
                    break

                images = Variable(images).type(torch.float).to(device)
                trans_emotion_id = Variable(trans_emotion_id).type(torch.long).to(device)
                mean_face = Variable(mean_face).type(torch.float).to(device)
                eigenfaces = Variable(eigenfaces).type(torch.float).to(device)

                # Face Discriminator
                FD_optim.zero_grad()
                logits_real = face_discriminator.forward(images)
                feat_latent = featCaptureNet.forward(images)
                fake_images = generator.forward(images, feat_latent, mean_face, eigenfaces).detach()
                logits_fake = face_discriminator.forward(fake_images)
                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_total_error.backward()        
                FD_optim.step()

                # Generator
                G_optim.zero_grad()
                feat_latent = featCaptureNet.forward(images)
                pred = generator.forward(images, feat_latent, mean_face, eigenfaces)
                gen_logits_fake = face_discriminator(pred)
                pred_labels = emotion_discriminator.forward(pred)

                _style_loss = loss_func.style_loss(images, pred)
                pred = pred.squeeze()
                images = images.squeeze()
                _l1_loss = l1_loss(pred,images)
                _l2_loss = l2_loss(pred,images)
                _emotion_loss = cross_entropy_loss(pred_labels, trans_emotion_id) 
                _face_loss = generator_loss(gen_logits_fake)

                loss = (_l1_loss + _l2_loss) + _emotion_loss + _style_loss + _face_loss 
                loss.backward()
                training_loss += (loss.item()/(len(train_1_loader)*7))
                G_optim.step()
        print('Epoch {}, training_type1_loss {:.3f}, time-elapse {:.3f}s'.format(e, training_loss, time.time()-start_t))


        # train on random expression (type 2)
        start_t = time.time()
        training_loss = 0.0
        for _, data in enumerate(train_2_loader):
            # Load input variable
            images, trans_emotion_id, _, mean_face, eigenfaces = data
            images = torch.unsqueeze(images, 1)

            if trans_emotion_id.shape[0] == 1:
                break

            images = Variable(images).type(torch.float).to(device)
            trans_emotion_id = Variable(trans_emotion_id).type(torch.long).to(device)
            mean_face = Variable(mean_face).type(torch.float).to(device)
            eigenfaces = Variable(eigenfaces).type(torch.float).to(device)

            # Face Discriminator
            FD_optim.zero_grad()
            logits_real = face_discriminator.forward(images)
            feat_latent = featCaptureNet.forward(images)
            fake_images = generator.forward(feat_latent, mean_face, eigenfaces).detach()
            logits_fake = face_discriminator.forward(fake_images)
            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()        
            FD_optim.step()

            # Generator
            G_optim.zero_grad()
            feat_latent = featCaptureNet.forward(images)
            pred = generator.forward(feat_latent, mean_face, eigenfaces)
            gen_logits_fake = face_discriminator(pred)
            pred_labels = emotion_discriminator.forward(pred)

            _style_loss = loss_func.style_loss(images, pred)
            pred = pred.squeeze()
            images = images.squeeze()
            _emotion_loss = cross_entropy_loss(pred_labels, trans_emotion_id) 
            _face_loss = generator_loss(gen_logits_fake)

            loss = _emotion_loss + _face_loss + _style_loss
            loss.backward()
            training_loss += (loss.item()/len(train_2_loader))
            G_optim.step()
        print('Epoch {}, training_type2_loss {:.3f}, time-elapse {:.3f}s'.format(e, training_loss, time.time()-start_t))

        # Save model:
        best_model['G_state_dict'] = generator.state_dict()
        best_model['FD_state_dict'] = face_discriminator.state_dict()
        save_name = save_dir + "/epoch_{}.pth".format(e)
        if (e+1)%10 == 0:
            torch.save(best_model, save_name)

def test(idx, config, epoch, save=True):
    # Using device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # Load dataset
    test_data = FER2013_EmotionTransform(dataset_type='test', training_type=2)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Models
    latent_features = 128
    featCaptureNet = FeatCaptureNet(latent_features=latent_features, only_latent=True)
    pretrain_faceCapture_ckpt_path = './checkpoints/featCaptureNet/no_skip_2/epoch_199.pth'
    featCaptureNet.load_state_dict(torch.load(pretrain_faceCapture_ckpt_path)['model'])
    featCaptureNet.to(device)
    featCaptureNet.eval()

    n_eigenfaces = 244
    generator = EmotionTransformNet(latent_features=latent_features, output_features=n_eigenfaces)
    pretrain_emotionTransform_ckpt_path = './checkpoints/EmotionTransformNet/{}/epoch_{}.pth'.format(config['name'], epoch)
    generator.load_state_dict(torch.load(pretrain_emotionTransform_ckpt_path)['G_state_dict'])
    generator.to(device)
    generator.eval()

    list_images = []
    list_labels = []
    list_pred_images = []

    test_case = [3,28,25,36]

    for batch_id, data in enumerate(test_loader):
        if batch_id not in test_case:
            continue
        if batch_id > 40:
            break
        # Load input variable
        images, trans_emotion_id, original_emotion, mean_face, eigenfaces = data
        images = torch.unsqueeze(images, 1)

        images = Variable(images).type(torch.float).to(device)
        trans_emotion_id = Variable(trans_emotion_id).type(torch.long).to(device)
        mean_face = Variable(mean_face).type(torch.float).to(device)
        eigenfaces = Variable(eigenfaces).type(torch.float).to(device)

        # Generator
        feat_latent = featCaptureNet.forward(images)
        pred = generator.forward(images, feat_latent, mean_face, eigenfaces)

        pred = pred.squeeze().detach().cpu().numpy()
        images = images.squeeze().detach().cpu().numpy()

        list_images.append(images)
        list_labels.append([original_emotion.item(), trans_emotion_id.item()])
        list_pred_images.append(pred)


    if save:
        fig, axes = plt.subplots(nrows=2, ncols=4)
        for i in range(4):

            axes[0, i].imshow(list_images[i], cmap='gray')
            axes[0, i].set(title='{}'.format(test_data.label_dict[list_labels[i][0]]))

            axes[1, i].imshow(list_pred_images[i], cmap='gray')
            axes[1, i].set(title='{}'.format(test_data.label_dict[list_labels[i][1]]))

        fig.suptitle('Top: Original, Bottom: Emotion Transformed')
        fig.savefig('emotion_transformed.png')
    return pred, images
        
if __name__ == "__main__":

    config = {
        'epochs': 200,
        'learning_rate': 5e-5,
        'name': 'final_2'
    }
    # train(config)

    pred, images = test(0, config, epoch=199)




