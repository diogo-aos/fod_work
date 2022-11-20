#!/usr/bin/env python
# coding: utf-8
import glob
import itertools
import logging


# In[ ]:

import mlflow
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch

# dataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
import time

import albumentations as albu

import dataclasses
import json

from core import Dataset, get_preprocessing, visualize, RunConfig
from utils import mlflow_log_eval as log_eval

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


train_sources = ['rpi', 'tx2']
test_sources = ['rpi_unseen', 'tx2_unseen']
all_sources = train_sources + test_sources


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
        im_dirs = [os.path.join(BASE_DIR, str(crop), name, 'images') for name in train_sources]
        seg_dirs = [os.path.join(BASE_DIR, str(crop), name, 'seg') for name in train_sources]
    elif dataset in all_sources:
        im_dirs = [os.path.join(BASE_DIR, str(crop), dataset, 'images')]
        seg_dirs = [os.path.join(BASE_DIR, str(crop), dataset, 'seg')]
    else:
        raise ValueError(f"dataset name must be 'merge' or one of {all_sources}")
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
    valid_loader = DataLoader(valid_dataset, batch_size=run_config.train_batch_size, shuffle=False, num_workers=4)


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

    
                # configure early stopping
                if valid_logs[run_config.patience_score] - patience_prev_score < run_config.patience_tolerance:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= run_config.patience_epochs:
                    # reached limit for epochs without relevant improvement
                    logging.info("early stop")
                    break

                patience_prev_score = valid_logs[run_config.patience_score]
            

            # test best model
            best_model = torch.load(model_path)
            
            # test set
            logging.info(f"evaluating on test set")
            # use test set to determine inference time
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
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


            # test on unseen images
            for dataset in test_sources:
                logging.info(f"evaluating on dataset {dataset}")
                data_dir = os.path.join(BASE_DIR, str(run_config.crop), dataset)
                logs = eval_dataset(
                    data_dir=data_dir,
                    encoder_name=run_config.encoder,
                    model=best_model,
                    classes=run_config.classes,
                    batch_size=run_config.train_batch_size,
                    loss_name=run_config.loss,
                    optimizer_name=run_config.optimizer,
                    store_predictions=True,
                    predictions_prefix=run_name,
                )
                log_eval(logs, prefix=f"test_{dataset}_")


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
        max_epochs=1,
        train_batch_size=16,
        eval_batch_size=1,
        dataset='tx2',  # rpi, tx2 or merge
        split=(0.8, 0.1, 0.1),
        crop=416
    )

    train(run_config=config)


def grid_search():
    config_gen = gridsearch.get_grid_search_generator(train_batch_size=64, max_epochs=200)
    for config in config_gen:
        while config.train_batch_size >= 1:
            print(json.dumps(dataclasses.asdict(config), indent=2))
            p = mp.Process(target=train, args=(config,))
            p.start()
            p.join()

            # if process didn't return normally, assume memory issue, reduce batch size and try again
            if p.exitcode != 0:
                config.train_batch_size = int(config.train_batch_size / 2)
            else:
                break


def visualize_write(out_fn, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()
    plt.savefig(out_fn)
    plt.close()


def predict(best_model, dataset, dataset_viz, viz_dir):
    ids = list(dataset.mask_ids.keys())
    for i in range(len(dataset)):       
        image_vis = dataset_viz[i][0].astype('uint8')
        image, gt_mask = dataset[i]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        visualize_write(
            os.path.join(viz_dir, f"{ids[i]}.png"),
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )

def eval_dataset(data_dir,
                 encoder_name,
                 model,
                 classes,
                 batch_size: int = 1,
                 loss_name="dice", optimizer_name="adam",
                 store_predictions: bool = False,
                 predictions_prefix: str = ''):
    # load preprocessing fun
    preprocessing_fn = get_preprocessing_fn(encoder_name=encoder_name)

    # load dataset
    im_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "seg")

    eval_dataset = Dataset(
            im_dir,
            mask_dir,
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=classes,
        )
    data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    if loss_name == 'dice':
        loss = smp.utils.losses.DiceLoss()
    else:
        raise ValueError(f'loss function selected {loss_name} is invalid')

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam([
            dict(params=model.parameters())
        ])
    else:
        raise ValueError(f'optimizer selected {optimizer_name} is invalid')


    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    test_epoch = smp.utils.train.ValidEpoch(
                            model,
                            loss=loss,
                            metrics=metrics,
                            device=DEVICE,
                            verbose=True,
                        )
    logs = test_epoch.run(data_loader)

    if store_predictions:
        viz_dir = os.path.join(data_dir, f"pred_{predictions_prefix}")
        os.makedirs(viz_dir, exist_ok=True)
        eval_dataset_viz = Dataset(
            im_dir,
            mask_dir,
            classes=classes,
        )
        predict(model, dataset=eval_dataset, dataset_viz=eval_dataset_viz, viz_dir=viz_dir)



    return logs


def eval_log_dataset(mlflow_run_id, model_path, base_dir, dataset_name: str, store_predicitons: bool = False):
    # get data from mlflow run
    run = mlflow.get_run(mlflow_run_id)

    # compute data_dirs based on dataset and crop
    data_dir = os.path.join(base_dir, str(run.data.params["crop"]), dataset_name)

    # parse classes
    classes = [c.strip("'") for c in run.data.params['classes'][1:-1].split(",")]
    #with mlflow.start_run(run_id=mlflow_run_id) as active_run:
    logs = eval_dataset(
        data_dir=data_dir,
        encoder_name=run.data.params["encoder"],
        model=torch.load(model_path),
        classes=classes,
        batch_size=int(run.data.params["train_batch_size"]),
        loss_name=run.data.params["loss"],
        optimizer_name=run.data.params["optimizer"],
        store_predictions=store_predicitons,
    )
    print(logs)
    #log_eval(logs, prefix=f"test_{dataset_name}_")

        

def eval_log_dataset_on_models(base_data_dir, store_predicitons: bool = False):
    # get all runs
    current_experiment=dict(mlflow.get_experiment_by_name(os.environ['MLFLOW_EXPERIMENT_NAME']))
    runs = mlflow.search_runs(current_experiment['experiment_id'])
    
    # list saved models and get the run_ids of those models
    models_fn_list = glob.glob(os.path.join(MODEL_OUTPUT_DIR, "*.pth"))
    models_run_names = [os.path.basename(fn).split('.')[0].split('best_model_')[-1] for fn in models_fn_list]
    run_names = pd.DataFrame()
    run_names["tags.mlflow.runName"] = models_run_names
    run_names["model_path"] = models_fn_list

    # runs that resulted in models
    runs = runs.merge(run_names, how="inner")

    run_ids = runs["run_id"]
    model_paths = runs["model_path"]


    # for each run, evaluate that model on the new test sets and log the results with mlflow
    for (run_id, model_path), dataset_name in itertools.product(zip(run_ids, model_paths), test_sources):
        print(dataset_name, run_id, model_path)
        eval_log_dataset(mlflow_run_id=run_id, model_path=model_path, base_dir=base_data_dir, dataset_name=dataset_name, store_predicitons=store_predicitons)



def fix_crop_param():
    ''' this functions sets the crop parameter to "original" on all mlflow runs that don't have this parameter set'''
    current_experiment=dict(mlflow.get_experiment_by_name(os.environ['MLFLOW_EXPERIMENT_NAME']))
    runs = mlflow.search_runs(current_experiment['experiment_id'])
    for run_id in runs['run_id']:
        run = mlflow.get_run(run_id=run_id)
        if 'crop' not in run.data.params:
            with mlflow.start_run(run_id=run_id):
                print(run_id)
                mlflow.log_param('crop', 'original')


def add_avg_unseen_iou():
    current_experiment=dict(mlflow.get_experiment_by_name(os.environ['MLFLOW_EXPERIMENT_NAME']))
    runs = mlflow.search_runs(current_experiment['experiment_id'])

    # filter failed runs
    runs = runs[runs["metrics.epoch"] != 0]

    # unssen iou score columns
    iou_cols = [col for col in runs.columns if "unseen_iou_score" in col and col != "avg_unseen_iou_score"]

    # compute avg unseen iou score
    runs["new_iou_score_unseen"] = runs[iou_cols].mean(axis=1)

    for _, row in runs.iterrows():
        run_id = row["run_id"]
        run = mlflow.get_run(run_id=run_id)
        
        if 'avg_unseen_iou_score' not in run.data.params:
            avg_score = row["new_iou_score_unseen"]
            print(run_id, avg_score)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric('avg_unseen_iou_score', avg_score)

if __name__ == '__main__':
    grid_search()
    #add_avg_unseen_iou()
    #main()
    #eval_log_dataset_on_models("data/crops", store_predicitons=True)
