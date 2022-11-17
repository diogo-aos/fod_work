# In[ ]:

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch

# dataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.data import Dataset as BaseDataset

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import albumentations as albu

import dataclasses

from crop import crop_dataset_core

from typing import List, Sequence, Optional




@dataclasses.dataclass()
class RunConfig:
    segmentation_model: str
    encoder: str
    encoder_weights: str
    classes: List[str]
    activation: str
    optimizer: str
    optimizer_lr: float
    loss: str
    max_epochs: int
    train_batch_size: int
    eval_batch_size: int
    dataset: int
    split: Sequence[float]
    split_seed: Optional[int] = None
    crop: str = 'original'
    patience_epochs: int = 15
    patience_tolerance: float = 0.01
    patience_score: str = 'iou_score'

# %% make dataset
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = {'fod': 255}

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        mask_fns = os.listdir(masks_dir)
        im_fns = os.listdir(images_dir)

        mask_ids = {fn.split('.')[0]: fn for fn in mask_fns}
        im_ids = {fn.split('.')[0]: fn for fn in im_fns}

        self.masks_fps = [os.path.join(masks_dir, fn) for _, fn in mask_ids.items()]
        self.images_fps = [os.path.join(images_dir, im_ids[id_]) for id_,_ in mask_ids.items()]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES[cls.lower()] for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.mask_ids = mask_ids

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # make image divisible by 32
        image = image[:1056, :, :]
        mask = mask[:1056, :, :]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        # return np.transpose(image[:1056,:,:], (2,0,1)), np.transpose(mask[:1056,:,:], (2,0,1))

    def __len__(self):
        return len(self.masks_fps)


class FilenameDataset(BaseDataset):
    def __init__(self, images_dir, masks_dir):
        raise NotImplemented()
        self.ids = []
        self.masks_fns = [os.path.join(masks_dir, f'{image_id}.png') for image_id in self.ids]
        self.images_fns = [os.path.join(images_dir, f'{image_id}.jpg') for image_id in self.ids]

    def __getitem__(self, i):
        return self.images_fns[i], self.masks_fns[i]

    def __len__(self):
        return len(self.ids)

class CroppedDataset(BaseDataset):
    def __init__(self, files_dataset: FilenameDataset,
                 crop_base_dir,
                 crop_size: int = 512) -> None:
        super().__init__()
        self.files_dataset = files_dataset

        
        # make dirs for cropped images and masks
        self.crop_pairs_fns = crop_dataset_core(files_dataset, crop_base_dir,
                                                crop_size=crop_size,
                                                filter_empty=True)

    def __getitem__(self, i):
        return self.crop_pairs_fns[i]

    def __len__(self):
        return len(self.crop_pairs_fns)



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()



