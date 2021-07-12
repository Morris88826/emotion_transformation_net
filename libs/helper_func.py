import numpy as np
import torch
import torch.nn as nn

def PCA(image, num_pc=10):
    u, s, vh = np.linalg.svd(image)

    sigma_ = np.diag(s[:num_pc])
    u_ = u[:,:num_pc]
    vh_ = vh[:num_pc, :]

    new_image = np.linalg.multi_dot([u_, sigma_, vh_])

    return new_image, (u_, sigma_, vh_)

def mean_face_generation(images):
    return (np.sum(images, axis=0)/images.shape[0]).astype(np.uint8)


def edge_detector(images):
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    weight = np.array([[0,1,0],[1,-4,1],[0,1,0]])[np.newaxis, np.newaxis, :]
    conv2d.weight = torch.nn.Parameter(torch.Tensor(weight))
    edges = conv2d(images)
    
    return edges