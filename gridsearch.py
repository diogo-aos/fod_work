from core import RunConfig
from itertools import product
from typing import Sequence


segmentation_models = [
    'unet',
    'DeepLabV3',
]

encoder_weights = [
#    ('resnet18', ['imagenet', 'ssl']),
    ('resnet34', ['imagenet']),
    ('resnet50', ['imagenet']), # ,'ssl', 'swsl'
#    ('resnet101', ['imagenet']),
#    ('resnet152', ['imagenet']),
#    ('timm-mobilenetv3_large_100', ['imagenet']),
]

optimizers = [
    'adam'
]

losses = [
    'dice'
]

activations = [
    'sigmoid'
]

learning_rate = [
    1e-3,
    1e-4
]

datasets = [
    'rpi',
    'tx2',
    'merge'
]


def get_grid_search_generator(train_batch_size: int = 8, max_epochs: int = 50, split: Sequence = (0.8, 0.1, 0.1)):
    for seg_model, (enc, weight_lst), optimizer, loss, activation, lr, dataset in \
         product(segmentation_models, encoder_weights, optimizers, losses, activations, learning_rate, datasets):
        for weights in weight_lst:
            yield RunConfig(
                segmentation_model=seg_model,
                encoder=enc,
                encoder_weights=weights,
                activation=activation,
                optimizer=optimizer,
                optimizer_lr=lr,
                loss=loss,
                dataset=dataset,
                max_epochs=max_epochs,
                train_batch_size=train_batch_size,
                eval_batch_size=1,
                split=split,
                classes=['fod'],
            )


if __name__ == '__main__':
    gen = get_grid_search_generator()
    lst = list(gen)
    print(len(lst), lst)