import torch
from torch.utils.data import Dataset
import os
import numpy as np
import skimage
from PIL import Image
import pandas as pd




class MIMICCXRDataset(Dataset):
    def __init__(self, dataframe, PATH_TO_IMAGES, transform=None):
        
        """
            Dataset class representing MIMIC-CXR dataset.
            This dataloader is compatible with MIMIC-CXR version 1. 
            
            Arguments:
            dataframe: Whether the dataset represents the train, test, or validation split
            PATH_TO_IMAGES: Path to the image directory on the server
            transform: Whether conduct transform to the images or not
            
            Returns:
            image, label and item["path"] as the unique indicator of each item in the dataloader.
        """        
        
        
        self.dataframe = dataframe
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.PATH_TO_IMAGES = PATH_TO_IMAGES

        self.PRED_LABEL = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Lesion',
            'Airspace Opacity',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]

        img = skimage.io.imread(os.path.join(self.PATH_TO_IMAGES, item["path"]))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(0, len(self.PRED_LABEL)):

            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')

        return img, label, item["path"]

    def __len__(self):
        return self.dataset_size
