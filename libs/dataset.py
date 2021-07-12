import pandas as pd
import numpy as np
import tqdm
import os
import pickle as pkl

class FER2013_dataset():
    def __init__(self, root = "./dataset/fer2013"):
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
        self.root = root  
        self.data = self.load_all_data()  
    
    def generate_all_data(self, datafile_path = "./dataset/fer2013/curated_fer2013.csv", out_path='./dataset/fer2013/all_data.pkl'):
        print('Generating all data ~')
    
        def _get_all_faces(df, emotion_id, size=self.size):

            data = df[df['emotion'] == emotion_id]
            data = data['pixels'].to_numpy()

            images = np.zeros((data.shape[0], size, size)).astype(np.uint8)

            for i in range(data.shape[0]):
                image = np.array([int(j) for j in data[i].split(' ')]).reshape((size, size))
                images[i] = image

            return images

        def _extract_data(df, usage):
            data = {}
            _df = df[df['Usage'] == usage]
            for emotion_id in tqdm.tqdm(self.label_dict.keys()):
                images = _get_all_faces(_df, emotion_id)
                data[emotion_id] = images
            return data

        df = pd.read_csv(datafile_path)

        all_data = {}
        # Train
        train_data = _extract_data(df, usage='Training')
        all_data['train'] = train_data
        # Val
        val_data = _extract_data(df, usage='PublicTest')
        all_data['val'] = val_data
        # Test
        test_data = _extract_data(df, usage='PrivateTest') 
        all_data['test'] = test_data
        
        with open(out_path, 'wb') as pickle_file:
            pkl.dump(all_data, pickle_file, protocol=pkl.HIGHEST_PROTOCOL)


    def load_all_data(self, datafile_path='./dataset/fer2013/all_data.pkl'):
        if not os.path.exists(datafile_path):
            self.generate_all_data()
        with open(datafile_path, 'rb') as pickle_file:
            data = pkl.load(pickle_file)
        
        return data

    def get_data(self, config):
        images = self.data[config["dataset_type"]][config["emotion_id"]]
        return images

# if __name__ == "__main__":
#     dataset = FER2013_dataset()
#     images = dataset.get_data('train', emotion_id=1)
