import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import tqdm
import os
import pickle as pkl
import glob
import random

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)

class FER2013_EmotionTransform(Dataset):
    def __init__(self, dataset_type, training_type=0, datafile_path = "./dataset/fer2013/curated_fer2013.csv"):
        
        super(FER2013_EmotionTransform, self).__init__()
        random.seed(0)

        self.label_dict = {
            0:'Angry',
            1:'Disgust',
            2:'Fear',
            3:'Happy',
            4:'Sad',
            5:'Surprise',
            6:'Neutral',
        }
        self.size = 48
        self.dataset_type = dataset_type
        self.datafile_path = datafile_path
        self.df = pd.read_csv(self.datafile_path)
        self.emotion_dataset = MeanFace_EigenFace()
        self.training_type = training_type

        if self.dataset_type == 'train':
            self.raw_images, self.labels = self.extract_data('Training')
        elif self.dataset_type == 'val':
            self.raw_images, self.labels = self.extract_data('PublicTest')
        elif self.dataset_type == 'test':
            self.raw_images, self.labels = self.extract_data('PrivateTest')
        else:
            print('Error dataset type!')
            raise NotImplementedError
        
        self.dataset_length = self.labels.shape[0]

    def __len__(self):

        if self.training_type == 1:
            return 7
        else:
            return self.dataset_length

    def __getitem__(self, idx):

        if self.training_type == 1:
            original_id = idx
        else:
            raw_image = self.raw_images[idx]
            original_id =self.labels[idx]
            image = np.array([float(i) for i in raw_image.split(' ')]).reshape((self.size, self.size))

        if self.training_type == 0: 
            emotion_id = original_id     
        else:
            emotion_id = random.randint(0, len(self.label_dict)-1)
            if idx == 12:
                emotion_id = 5
            elif idx == 25:
                emotion_id = 5
            elif idx == 22:
                emotion_id = 2
            elif idx == 38:
                emotion_id = 3
        
        mean_face, eigen_faces = self.emotion_dataset.get_emotion_info(emotion_id)
        mean_face = mean_face.flatten()
        eigen_faces = eigen_faces.reshape((-1, 48*48))

        if self.training_type == 1:
            image, _ = self.emotion_dataset.get_emotion_info(idx)
        else:
            image = image/255.0

        return image, emotion_id, original_id, mean_face, eigen_faces

    def normalized(self, image):
        n_image = (image - np.mean(image))/(np.amax(image)-np.amin(image))
        return n_image

    def extract_data(self, usage):
        data = self.df[self.df['Usage'] == usage]
        emotion_labels = data['emotion'].to_numpy()
        raw_images = data['pixels'].to_numpy()
        return raw_images, emotion_labels

class MeanFace_EigenFace():
    def __init__(self, file_path='./dataset/fer2013/meanfaces_eigenfaces.pkl'):
        with open(file_path, 'rb') as pickle_file:
            self.data = pkl.load(pickle_file)

    def get_emotion_info(self, emotion):
        mean_face = self.data[emotion]['mean_face']
        eigen_faces = self.data[emotion]['eigenfaces']
        return mean_face, eigen_faces

class FER2013_loader(Dataset):
    def __init__(self, dataset_type, datafile_path = "./dataset/fer2013/curated_fer2013.csv"):
        
        super(FER2013_loader, self).__init__()

        self.label_dict = {
            0:'Angry',
            1:'Disgust',
            2:'Fear',
            3:'Happy',
            4:'Sad',
            5:'Surprise',
            6:'Neutral',
        }
        self.size = 48
        self.dataset_type = dataset_type
        self.datafile_path = datafile_path
        self.df = pd.read_csv(self.datafile_path)

        if self.dataset_type == 'train':
            self.raw_images, self.labels = self.extract_data('Training')
        elif self.dataset_type == 'val':
            self.raw_images, self.labels = self.extract_data('PublicTest')
        elif self.dataset_type == 'test':
            self.raw_images, self.labels = self.extract_data('PrivateTest')
        else:
            print('Error dataset type!')
            raise NotImplementedError
        
        self.dataset_length = self.labels.shape[0]

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        raw_image = self.raw_images[idx]
        image = np.array([float(i) for i in raw_image.split(' ')]).reshape((self.size, self.size))
        
        if (np.amax(image)-np.amin(image)) == 0:
            print(self.raw_images[idx])
            raise NotImplementedError
        # n_image = self.normalized(image)
        image = image/255.0
        label = self.labels[idx]
        return image, label

    def normalized(self, image):
        n_image = (image - np.mean(image))/(np.amax(image)-np.amin(image))
        return n_image
    
    def extract_data(self, usage):
        data = self.df[self.df['Usage'] == usage]
        emotion_labels = data['emotion'].to_numpy()
        raw_images = data['pixels'].to_numpy()
        return raw_images, emotion_labels
    
    def overview(self):
        print('Dataset: FER2013, {}'.format(self.dataset_type))
        print('Image Size: 48x48')
        print('Number of samples: ', self.labels.shape[0])
        print('Data distribution:')
        hist, _ = self.get_histogram()
        for i in range(hist.shape[0]):
            print("{}: {}".format(self.label_dict[i],hist[i]))

    def get_histogram(self):
        hist, bin_edges = np.histogram(self.labels, bins=np.arange(len(self.label_dict)+1))
        return hist, bin_edges

class FEC_loader(Dataset):
    def __init__(self, root = "./dataset/FEC_dataset/images"):
        super(FEC_loader, self).__init__()
        self.root = root
        self.dataset_length = len(glob.glob(root+'/*'))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        image_path = "{}/{:05d}.png".format(self.root,idx)
        image = np.array(Image.open(image_path).resize((48,48)))
        n_image = self.normalized(image)
        return n_image
        
    def normalized(self, image):
        n_image = (image - np.mean(image))/(np.amax(image)-np.amin(image))
        return n_image
    
def generate_FEC_data(datafile_path="./dataset/FEC_dataset/faceexp-comparison-data-train-public.csv", output_folder='./dataset/FEC_dataset/images'):
    col_names = ["URL1", "TopLeftColumn1", "BottomRightColumn1", 'TopLeftRow1', 'BottomRightRow1',
                "URL2", "TopLeftColumn2", "BottomRightColumn2", 'TopLeftRow2', 'BottomRightRow2',
                "URL3", "TopLeftColumn3", "BottomRightColumn3", 'TopLeftRow3', 'BottomRightRow3',
                "Triplet_type", "AnnotatorID1", "Annotation1", "AnnotatorID2", "Annotation2", "AnnotatorID3", "Annotation3", "AnnotatorID4", "Annotation4", "AnnotatorID5", "Annotation5", "AnnotatorID6", "Annotation6"]
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    df = pd.read_csv(datafile_path, names=col_names)

    URL1 = df['URL1'].to_list()
    URL2 = df['URL2'].to_list()
    URL3 = df['URL3'].to_list()

    URL = list(set(URL1 + URL2 + URL3))

    image_idx = 0
    for i in tqdm.tqdm(range(len(URL))):
        _idx = 1
        data = df[df['URL1'] == URL[i]]
        if len(data) == 0:
            _idx = 2
            data = df[df['URL2'] == URL[i]]
            if len(data) == 0:
                _idx = 3
                data = df[df['URL3'] == URL[i]]
                if len(data) == 0:
                    raise NotImplementedError

        data = data.loc[:, ['URL{}'.format(_idx), 'TopLeftColumn{}'.format(_idx), 'BottomRightColumn{}'.format(_idx), 'TopLeftRow{}'.format(_idx), 'BottomRightRow{}'.format(_idx)]].to_numpy()[0]
        url, start_x, end_x, start_y, end_y = list(data)
        
        try:
            response = requests.get(url)
            img = np.array(Image.open(BytesIO(response.content)))
        except:
            print(url)
            continue

        if img.ndim == 3:
            h, w, c = img.shape
            img = rgb2gray(img)
        else:
            h, w = img.shape

        
        
        start_x = int(w*start_x)
        end_x = int(w*end_x)
        start_y = int(h*start_y)
        end_y = int(h*end_y)
        length = max(end_x-start_x, end_y-start_y)

        end_x = min(start_x+length, w)
        end_y = min(start_y+length, w)

        s_img = img[start_y:end_y, start_x:end_x]
        
        new_image = np.zeros((length, length),dtype=np.uint8)
        new_image[:s_img.shape[0], :s_img.shape[1]] = s_img

        resized_img = Image.fromarray(new_image).resize((128,128)).save('{}/{:05d}.png'.format(output_folder,image_idx),"PNG")
        image_idx += 1

def organize_FER():
    datafile_path = "./dataset/fer2013/fer2013.csv"
    df = pd.read_csv(datafile_path).to_numpy()

    count = []
    for i in range(len(df)):
        raw_image = (df[i][1])
        image = np.array([float(i) for i in raw_image.split(' ')]).reshape((48, 48))
        if (np.amax(image)-np.amin(image)) == 0:
            count.append(i)
    
    new_df = pd.read_csv(datafile_path).drop(count)
    new_df = new_df.reset_index(drop=True)
    new_df.to_csv('./dataset/fer2013/curated_fer2013.csv')

if __name__ == "__main__":
    dataset= FER2013_loader('train')
    dataloader = DataLoader(dataset, batch_size=10)
    images, labels = next(iter(dataloader))

    images = torch.autograd.Variable(images)
    images = images.type(torch.float)

    processed = edge_detector(images.unsqueeze(1))
    processed = processed.squeeze().detach().cpu().numpy()
    
    import matplotlib.pyplot as plt
    plt.imshow(processed[2], cmap='gray')
    plt.savefig('edge.png')
    

    # fec_dataset = FEC_loader()
    # fec_loader = DataLoader(fec_dataset, batch_size=10)

    # images = next(iter(fec_loader))
    # print(images.shape)
