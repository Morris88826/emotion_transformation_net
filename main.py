import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import  DataLoader
from libs.helper_func import PCA
from libs.dataloader import FER2013_loader
import pickle as pkl

if __name__ == "__main__":
    
    test_data = FER2013_loader(dataset_type='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # Projection Preview
    list_images = []
    list_labels = []
    t = 0
    for data in test_loader:
        images, labels = data
        images = images.squeeze().detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()

        if labels.item() == t:
            list_images.append(images)
            list_labels.append(labels.item())
            t += 1
            
            if t == 7:
                break
    with open("./dataset/fer2013/meanfaces_eigenfaces.pkl", 'rb') as pkl_file:
        eigen_data = pkl.load(pkl_file)

    fig, axes = plt.subplots(nrows=2, ncols=4)
    for i in range(4):
        image = list_images[i]
        label = list_labels[i]

        mean_face = eigen_data[i]['mean_face'].flatten()
        eigenfaces = eigen_data[i]['eigenfaces'].reshape((-1, 48*48))
        emotion_data = image.reshape((-1, 48*48))
        normalized_data = emotion_data - mean_face
        recontructed = np.copy(mean_face)
        for j in range(eigenfaces.shape[0]):
            weight = np.dot(normalized_data, eigenfaces[j])
            recontructed += weight*eigenfaces[j]

        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set(title="{}".format(test_data.label_dict[label]))

        axes[1, i].imshow(recontructed.reshape((48,48)), cmap='gray')
        axes[1, i].set(title="{}".format(test_data.label_dict[label]))
    fig.suptitle('Top: Original, Bottom: Reconstructed (num_pc: 244)')
    fig.savefig('projection.png')

    # Dataset Preview
    # fig, axes = plt.subplots(nrows=2, ncols=5)
    # for i in range(images.shape[0]):
    #     axes[i//5, i%5].imshow(images[i], cmap='gray')
    #     axes[i//5, i%5].set(title=train_data.label_dict[labels[i]])
    
    # fig.tight_layout()
    # fig.savefig('dataset.png')

    # Mean Face Preview
    # fig, axes = plt.subplots(nrows=2, ncols=4)
    # with open('./dataset/fer2013/meanfaces_eigenfaces.pkl', 'rb') as pkl_file:
    #     data = pkl.load(pkl_file)
    # for i in range(7):
    #     axes[i//4, i%4].imshow(data[i]['mean_face'], cmap='gray')
    #     axes[i//4, i%4].set(title=train_data.label_dict[i])
    # fig.tight_layout()
    # fig.savefig('meanface.png')