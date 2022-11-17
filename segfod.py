#!/usr/bin/env python
# coding: utf-8
import logging

# In[ ]:

import mlflow
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch

# dataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

import os
import numpy as np
import time

import albumentations as albu

import dataclasses
import json

from core import Dataset, get_preprocessing, visualize, RunConfig

import os
import sys
import multiprocessing as mp

import gridsearch

import random

from typing import Tuple, Optional, Union

# fail if tracking URI not set
os.environ['MLFLOW_TRACKING_URI']
os.environ['MLFLOW_EXPERIMENT_NAME']

# In[2]:


#%%

BASE_DIR = os.environ['CROPS_OUTPUT_DIR']
MODEL_OUTPUT_DIR = os.environ['MODEL_OUTPUT_DIR']


sources = ['rpi', 'tx2']

DEVICE = 'cuda'


# set random seed
GLOBAL_SEED = 67280421310721
#random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
#np.random.seed(GLOBAL_SEED)


# check paths exist
for path in [MODEL_OUTPUT_DIR]:
    if not os.path.exists(path):
        raise FileExistsError(path)


def get_data_dirs(dataset, crop: Union[int, str] = 'original'):
    if dataset == 'merge':
        im_dirs = [os.path.join(BASE_DIR, str(crop), name, 'images') for name in sources]
        seg_dirs = [os.path.join(BASE_DIR, str(crop), name, 'seg') for name in sources]
    elif dataset in sources:
        im_dirs = [os.path.join(BASE_DIR, str(crop), dataset, 'images')]
        seg_dirs = [os.path.join(BASE_DIR, str(crop), dataset, 'seg')]
    else:
        raise ValueError(f"dataset name must be merge or one of {sources}")
    return im_dirs, seg_dirs


def train(run_config: RunConfig):

    if run_config.segmentation_model == 'unet':
        model = smp.Unet(
            encoder_name=run_config.encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=run_config.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(run_config.classes),                      # model output channels (number of classes in your dataset)
            activation=run_config.activation,
        )
    elif run_config.segmentation_model == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=run_config.encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=run_config.encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(run_config.classes),  # model output channels (number of classes in your dataset)
            activation=run_config.activation,
        )
    else:
        raise ValueError(f"segmentation model specified {run_config.segmentation_model} invalid")

    preprocessing_fn = get_preprocessing_fn(run_config.encoder, pretrained=run_config.encoder_weights)


    # load dataset
    im_dirs, mask_dirs = get_data_dirs(run_config.dataset, crop=run_config.crop)
    datasets = []
    for im_dir, mask_dir in zip(im_dirs, mask_dirs):
        new_dataset = Dataset(
            im_dir,
            mask_dir,
            #augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=run_config.classes,
        )
        datasets.append(new_dataset)
    final_dataset = ConcatDataset(datasets)
    

    random_generator = torch.Generator()
    random_generator.manual_seed(GLOBAL_SEED)
    run_config.split_seed = random_generator.initial_seed()

    train_dataset, valid_dataset, test_dataset = random_split(final_dataset, run_config.split, generator=random_generator)


    train_loader = DataLoader(train_dataset, batch_size=run_config.train_batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=run_config.eval_batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=run_config.eval_batch_size, shuffle=False, num_workers=4)


    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    if run_config.loss == 'dice':
        loss = smp.utils.losses.DiceLoss()
    else:
        raise ValueError(f'loss function selected {run_config.loss} is invalid')

    if run_config.optimizer == 'adam':
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=run_config.optimizer_lr),
        ])
    else:
        raise ValueError(f'optimizer selected {run_config.optimizer} is invalid')


    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]


    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,

    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    def log_eval(logs: dict, prefix: str = '', step: Optional[int] = None):
        for k, v in logs.items():
            mlflow.log_metric(f'{prefix}{k}', v, step=step)

    patience_counter = 0
    patience_prev_score = 0



    try:
        with mlflow.start_run() as run:
            run_name = run.data.tags['mlflow.runName']
            
            mlflow.log_params(dataclasses.asdict(run_config))
            mlflow.log_param('train_size', len(train_dataset))
            mlflow.log_param('validation_size', len(valid_dataset))
            mlflow.log_param('test_size', len(test_dataset))

            model_path = os.path.join(MODEL_OUTPUT_DIR, f'./best_model_{run_name}.pth')
            mlflow.log_param('best_model_path', model_path)

            max_score = 0
            for i in range(0, run_config.max_epochs):
                mlflow.log_metric('epoch', i, step=i)

                logging.info('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)

                log_eval(train_logs, prefix='train_', step=i)
                log_eval(valid_logs, prefix='val_   ', step=i)

                # do something (save model, change lr, etc.)
                if max_score < valid_logs['iou_score']:
                    max_score = valid_logs['iou_score']
                    

                    torch.save(model, model_path)

                    mlflow.log_artifact(local_path=model_path)
                    
                    logging.info('Model saved!')

                if i == 25:
                    optimizer.param_groups[0]['lr'] = run_config.optimizer_lr / 10  # reduce learning rate
                    logging.info('Decrease decoder learning rate to 1e-5!')

                # configure early stopping
                if valid_logs[run_config.patience_score] - patience_prev_score < run_config.patience_tolerance:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= run_config.patience_epochs:
                    # reached limit for epochs without relevant improvement
                    print("early stop")
                    break

                patience_prev_score = valid_logs[run_config.patience_score]
            
            best_model = torch.load(model_path)
            
            test_epoch = smp.utils.train.ValidEpoch(
                            best_model,
                            loss=loss,
                            metrics=metrics,
                            device=DEVICE,
                            verbose=True,
                        )
            
            # run best model on test set
            # measure inference time
            t_start = time.time()
            test_logs = test_epoch.run(test_loader)
            t_elapsed = time.time() - t_start

            mlflow.log_metric('inference_time_per_sample', t_elapsed / len(test_dataset))
            log_eval(test_logs, prefix='test_')

    except torch.cuda.OutOfMemoryError:
        logging.warning(f'cuda out of memory, batch size: {run_config.train_batch_size}')
        sys.exit(42)


def main():
    config = RunConfig(
        segmentation_model='unet',
        encoder="resnet18",
        encoder_weights='imagenet',
        classes=['fod'],
        activation='sigmoid',
        optimizer='adam',
        optimizer_lr=0.0001,
        loss='dice',
        max_epochs=50,
        train_batch_size=32,
        eval_batch_size=1,
        dataset='rpi',  # rpi, tx2 or merge
        split=(0.8, 0.1, 0.1),
        crop=416
    )

    train(run_config=config)


def grid_search():
    config_gen = gridsearch.get_grid_search_generator(train_batch_size=64)
    for config in config_gen:
        while config.train_batch_size >= 1:
            print(json.dumps(dataclasses.asdict(config), indent=2))
            p = mp.Process(target=train, args=(config,))
            p.start()
            p.join()
            if p.exitcode == 42:
                config.train_batch_size = int(config.train_batch_size / 2)
            else:
                break
        

if __name__ == '__main__':
    grid_search()
    #main()
