from typing import Optional, Tuple
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from datamodules.transforms import *

from datamodules.components.hecktor_dataset import HecktorDataset
import pandas as pd

from torch.utils.data.dataloader import default_collate
import torch
import copy
import pickle

isKFold = False
class myZeroPadding:
    def __init__(self, target_shape, mode='train'):
        self.target_shape = np.array(target_shape)  # without channel dimension
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        img, mask = sample['input'], sample['target_mask']
        input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored

        d_x, d_y, d_z = self.target_shape - input_shape
        d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)
        
        if not all(i == 0 for i in (d_x, d_y, d_z)):
            positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
            negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

            # padding for positive values:
            img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))
            mask = np.pad(mask, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

            # cropping for negative values:
            img = img[: negative[0], : negative[1], : negative[2], :].copy()
            mask = mask[: negative[0], : negative[1], : negative[2], :].copy()

            assert img.shape[:-1] == mask.shape[:-1], f'Shape mismatch for the image {img.shape[:-1]} and mask {mask.shape[:-1]}'

            sample['input'], sample['target_mask'] = img, mask

        return sample
class HECKTORDataModule():
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(self, data_dir='/l/users/thao.nguyen/cropped_dataset/', clinical_data_path='/l/users/thao.nguyen/tmss_miccai/', batch_size=4, num_workers=1):#Change


        self.data_dir = data_dir
        self.clinical_data_path = clinical_data_path          
        self.Fold = 1
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = True
        
        self.test_transforms = transforms.Compose(
            [ 
             NormalizeIntensity(),
            myZeroPadding(target_shape=np.array([96, 96, 96]),mode='train'), #change
              ToTensor()
              ]
        )

        self.train_transforms = transforms.Compose([
            NormalizeIntensity(),
            myZeroPadding(target_shape=np.array([96, 96, 96]),mode='train'), #change
            ToTensor(),
        ])



        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def data(self):

        self.dataset = HecktorDataset(#self.hparams["root_dir"],
                                      clinical_data_path=self.clinical_data_path, # directory to 2 .csv files
                                      #self.hparams["patch_sz"],
                                      time_bins = 14,
                                      transform=self.train_transforms,
                                      cache_dir=self.data_dir,
                                      num_workers=self.num_workers)
        

        # if isKFold:
        #     df = copy.copy(self.dataset.clinical_data)
        #     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5820222)

        #     train_idx = {}
        #     test_idx = {}

        #     key = 0
        #     for i,j in kf.split(df,df['event']):
        #         train_idx[key] = i
        #         test_idx[key] = j
        #         key += 1
        #     train_dataset = Subset(self.dataset, train_idx[0])
        #     val_dataset = Subset(self.dataset, test_idx[0])
        # else:
        train_size = int(0.8 * self.dataset.__len__())
        val_size = self.dataset.__len__() - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

    
        train_dataset.dataset.transform = self.train_transforms
        val_dataset.dataset.transform = self.test_transforms
    

        self.data_train = train_dataset
        self.data_val = val_dataset
       
        train_dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,)
        val_dataloader = DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True)
        return train_dataloader, val_dataloader


# Open the file in binary write mode and save the object
# filename = '/l/users/thao.nguyen/tmss_miccai/train_loader.pkl'
# with open(filename, 'wb') as file:
#     pickle.dump(train_loader, file)

# filename = '/l/users/thao.nguyen/tmss_miccai/val_loader.pkl'
# with open(filename, 'wb') as file:
#     pickle.dump(val_loader, file)


# train_loader = iter(train_loader)
# for i in range(len(train_loader)):
#     data = next(train_loader)
#     (sample, clin_var), target, labels = data
#     print(clin_var.shape)
#     break