import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from PIL import Image

class IntelDataset(Dataset):
    """PyTorch dataset class with transformers from Albumentations"""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        image = self.data.image[idx]
        label = self.data.labels[idx]
        
        image = Image.open(image)
        image = image.convert('RGB')
        image = np.array(image)
        
        if self.transform is not None:
            aug = self.transform(image=image)
            image = aug['image']

        return image, label


def create_dataframe(path, classes):
    """
    Creates a dataframe from a dataset directory.
    
    Assumes the structure:
    dataset/ 
    ├── class1/
    │   ├── img1_class1.jpg
    │   ├── img2_class1.jpg
    │   └── ...
    ├── class2/
    │   ├── img1_class2.jpg
    │   ├── img2_class2.jpg
    │   └── ...
    └── ...
    """
    labels, images = [], []
        
    for folder in os.listdir(path):
        # drop?
        if folder not in classes:
            continue
        label = classes[folder]
        path = os.path.join(path, folder)
        
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                labels.append(label)
                images.append(img_path)
    
    df = pd.DataFrame({'image': images, 'labels': labels})
    return shuffle(df).reset_index(drop=True)


def load_data(cfg):
    """Get data..."""
    label2id = {class_name: i for i, class_name in enumerate(cfg.classes)}

    train_set = create_dataframe(cfg.train_dir, label2id)
    test_set = create_dataframe(cfg.test_dir, label2id)

    train = IntelDataset(train_set, cfg.train_transform)
    test = IntelDataset(test_set, cfg.test_transform)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=cfg.batch_size, shuffle=True)
    return train_loader, test_loader