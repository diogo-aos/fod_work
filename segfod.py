#!/usr/bin/env python
# coding: utf-8

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

import albumentations as albu

import dataclasses

from core import Dataset, get_preprocessing, visualize, RunConfig



# In[2]:


#%%
RPI_IM_DIR = '/home/dasilva/fod_dataset/original/19dez_rpi_fod'
RPI_MASK_DIR = '/home/dasilva/fod_dataset/seg_maps/19dez_rpi_fod'

TX2_IM_DIR = '/home/dasilva/fod_dataset/original/19dez_tx2_fod'
TX2_MASK_DIR = '/home/dasilva/fod_dataset/seg_maps/19dez_tx2_fod'

DEVICE = 'cuda'

run_config = RunConfig(
    segmentation_model='unet',
    encoder="resnet101",
    encoder_weights='imagenet',
    classes=['fod'],
    activation='sigmoid',
    optimizer='adam',
    optimizer_lr=0.0001,
    loss='dice',
    max_epochs=50,
    train_batch_size=1,
    eval_batch_size=1,
    dataset='rpi',  # rpi, tx2 or merge
    split=(0.8,0.1,0.1)
)


#%%
dataset = Dataset(RPI_IM_DIR, RPI_MASK_DIR, classes=['fod'])


image, mask = dataset[1] # get some sample
visualize(
    image=image,
    fod_mask=mask.squeeze(),
)


# # Train

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


# In[ ]:


rpi_dataset = Dataset(
    RPI_IM_DIR,
    RPI_MASK_DIR,
    #augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=run_config.classes,
)

tx2_dataset = Dataset(
    TX2_IM_DIR,
    TX2_MASK_DIR,
    #augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=run_config.classes,
)

if run_config.dataset == 'rpi':
    final_dataset = rpi_dataset
elif run_config.dataset == 'tx2':
    final_dataset = tx2_dataset
elif run_config.dataset == 'merge':
    final_dataset = ConcatDataset((rpi_dataset, tx2_dataset))
else:
    raise ValueError(f"dataset must be either (rpi, tx2, merge)")

random_generator = torch.Generator()
run_config.split_seed = random_generator.initial_seed()
train_dataset, valid_dataset, test_dataset = random_split(final_dataset, run_config.split, generator=random_generator)


train_loader = DataLoader(train_dataset, batch_size=run_config.train_batch_size, shuffle=True, num_workers=8, generator=random_generator)
valid_loader = DataLoader(valid_dataset, batch_size=run_config.eval_batch_size, shuffle=False, num_workers=4, generator=random_generator)
test_loader = DataLoader(test_dataset, batch_size=run_config.eval_batch_size, shuffle=False, num_workers=4, generator=random_generator)

# In[ ]:



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


# In[ ]:


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


# In[ ]:


# train model for 40 epochs

def log_eval(logs: dict, prefix: str = '', step: int = 0):
    for k, v in logs.items():
        mlflow.log_metric(f'{prefix}{k}', v, step=step)


with mlflow.start_run() as run:
    run_name = run.data.tags['mlflow.runName']
    mlflow.log_params(dataclasses.asdict(run_config))
    max_score = 0

    for i in range(0, run_config.max_epochs):
        mlflow.log_metric('epoch', i, step=i)

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        log_eval(train_logs, prefix='train_', step=i)
        log_eval(valid_logs, prefix='val_   ', step=i)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            model_path = os.path.join('artifacts', f'./best_model_{run_name}.pth')
            torch.save(model, model_path)
            mlflow.log_artifact(local_path=model_path)
            mlflow.log_param('best_model_path', )
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


# # Load best model

# In[19]:


# load best saved checkpoint
best_model = torch.load('./best_model.pth')


# In[25]:


valid_epoch = smp.utils.train.ValidEpoch(
    best_model, 
    loss=loss,
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


# In[26]:


valid_logs = valid_epoch.run(valid_loader)


# # Visualize

# In[20]:


viz_valid_dataset = Dataset(VAL_IM_DIR, VAL_MASK_DIR, classes=['fod'])


# In[21]:
for i in range(5):
    n = np.random.choice(len(valid_dataset))
    
    image_vis = viz_valid_dataset[n][0].astype('uint8')
    image, gt_mask = valid_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )

