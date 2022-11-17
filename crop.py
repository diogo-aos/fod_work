import torchvision.transforms as transforms
import numpy as np
import torch
import random
import subprocess as sp
import os
import cv2
import glob
import numpy as np
import itertools
import tqdm



CROP_SIZE = [
    #256,
    416,
    512,
    #704,
    832,
    #960,
]

OVERLAP = 0.5


def get_env_params():
    RPI_IM_DIR = os.environ['RPI_IMAGE_DIR']
    RPI_SEG_DIR = os.environ['RPI_SEG_DIR'] 
    RPI_ANNOT_FN = os.environ['RPI_ANNOT_FN']

    TX2_IM_DIR = os.environ['TX2_IMAGE_DIR']
    TX2_SEG_DIR = os.environ['TX2_SEG_DIR']
    TX2_ANNOT_FN = os.environ['TX2_ANNOT_FN']

    UNSEEN_RPI_IMAGE_DIR = os.environ["UNSEEN_RPI_IMAGE_DIR"]
    UNSEEN_RPI_SEG_DIR = os.environ["UNSEEN_RPI_SEG_DIR"]

    UNSEEN_TX2_IMAGE_DIR = os.environ["UNSEEN_TX2_IMAGE_DIR"]
    UNSEEN_TX2_SEG_DIR = os.environ["UNSEEN_TX2_SEG_DIR"]


    OUTPUT_BASE_DIR = os.environ['CROPS_OUTPUT_DIR']

    sources = {
        #'rpi': (RPI_IM_DIR, RPI_SEG_DIR),
        #'tx2': (TX2_IM_DIR, TX2_SEG_DIR),
        'rpi_unseen': (UNSEEN_RPI_IMAGE_DIR, UNSEEN_RPI_SEG_DIR),
        'tx2_unseen': (UNSEEN_TX2_IMAGE_DIR, UNSEEN_TX2_SEG_DIR)
    }

    params = {
        'sources': sources,
        'OUTPUT_BASE_DIR': OUTPUT_BASE_DIR
    }

    return params

def main(filter_empty: bool):
    env_params = get_env_params()
    sources = env_params['sources']
    OUTPUT_BASE_DIR = env_params['OUTPUT_BASE_DIR']

    params = itertools.product(CROP_SIZE, sources.keys())
    for crop_size, dataset in params:
        print(crop_size, dataset)
        im_dir, seg_dir = sources[dataset]

        crop_dataset(im_dir=im_dir,
                    seg_dir=seg_dir,
                    out_dir=os.path.join(OUTPUT_BASE_DIR, str(crop_size), dataset),
                    crop_size=crop_size,
                    overlap=OVERLAP,
                    filter_empty=filter_empty)


def crop_dataset(im_dir, seg_dir, out_dir, crop_size, overlap, filter_empty: bool = True):
    # set and create output directories
    crop_seg_dir = os.path.join(out_dir, "seg")
    crop_im_dir = os.path.join(out_dir, "images")

    os.makedirs(crop_seg_dir, exist_ok=True)
    os.makedirs(crop_im_dir, exist_ok=True)

    seg_fns = glob.glob(os.path.join(seg_dir, "*"))
    seg_id_fns = {os.path.basename(fn).split('.')[0]: fn for fn in seg_fns}

    im_fns = glob.glob(os.path.join(im_dir, "*"))
    im_id_fns = {os.path.basename(fn).split('.')[0]: fn for fn in im_fns}

    n_deleted = 0
    n_crops = 0
    for seg_id, seg_fn in tqdm.tqdm(seg_id_fns.items()):
        # get corresponding original image path
        im_fn = im_id_fns[seg_id]

        # load original image and segmentation
        im_im = cv2.imread(im_fn)
        seg_im = cv2.imread(seg_fn)

        # if input segmentation is empty, then no area of interest exists, skip image
        if filter_empty:
            if seg_im.sum() == 0:
                continue

        # get crops
        im_crops = crop_image(im_im, crop_size=crop_size, overlap=overlap)
        seg_crops = crop_image(seg_im, crop_size=crop_size, overlap=overlap)

        # write crops
        
        for i, (seg_crop, im_crop) in enumerate(zip(seg_crops, im_crops)):
            # filter_empty is set and sum of all segmentation pixels is 0, then no annotation is present
            if filter_empty:
                if seg_crop.sum() == 0:
                    n_deleted += 1
                    continue

            # write crops
            out_seg_crop_fn = os.path.join(crop_seg_dir, f"{seg_id}_{i}.png")
            out_im_crop_fn = os.path.join(crop_im_dir, f"{seg_id}_{i}.png")
            cv2.imwrite(out_seg_crop_fn, seg_crop)
            cv2.imwrite(out_im_crop_fn, im_crop)
            n_crops += 1

    print(f'total crops: {n_crops}\tremoved: {n_deleted}')


def crop_dataset_core(image_mask_pairs_fns, out_dir, crop_size, overlap, filter_empty: bool = True):
    # set and create output directories
    crop_seg_dir = os.path.join(out_dir, "seg")
    crop_im_dir = os.path.join(out_dir, "images")

    os.makedirs(crop_seg_dir, exist_ok=True)
    os.makedirs(crop_im_dir, exist_ok=True)

    n_deleted = 0
    n_crops = 0

    crop_fns = []  # return list

    # iterate over all (mask, image) pair
    for im_fn, seg_fn in tqdm.tqdm(image_mask_pairs_fns):
        crop_id = os.path.basename(im_fn).split('.')[0]

        # load original image and segmentation
        im_im = cv2.imread(im_fn)
        seg_im = cv2.imread(seg_fn)

        # if input segmentation is empty, then no area of interest exists, skip image
        if filter_empty:
            if seg_im.sum() == 0:
                continue

        # get crops
        im_crops = crop_image(im_im, crop_size=crop_size, overlap=overlap)
        seg_crops = crop_image(seg_im, crop_size=crop_size, overlap=overlap)

        # write crops
        
        for i, (seg_crop, im_crop) in enumerate(zip(seg_crops, im_crops)):
            # filter_empty is set and sum of all segmentation pixels is 0, then no annotation is present
            if filter_empty:
                if seg_crop.sum() == 0:
                    n_deleted += 1
                    continue

            # write crops
            cv2.imwrite((out_seg_crop_fn := os.path.join(crop_seg_dir, f"{crop_id}_{i}.png")),
                        seg_crop)
            cv2.imwrite((out_im_crop_fn := os.path.join(crop_im_dir, f"{crop_id}_{i}.png")),
                        im_crop)

            crop_fns.append(out_im_crop_fn, out_seg_crop_fn)
            n_crops += 1

    print(f'total crops: {n_crops}\tremoved: {n_deleted}')
    return crop_fns


def crop_image(im: np.ndarray, crop_size: int = (512, 512), overlap: float = 0.1):
    if isinstance(crop_size, tuple):
        cw, ch = crop_size 
    elif isinstance(crop_size, int):
        cw = ch = crop_size
    else:
        raise TypeError(f'type of crop size must be tuple (width, height) or int')

    h, w, _ = im.shape

    # if crop size bigger than image, return image
    if cw >= w and ch >= h:
        return [im]
    
    # overlap in pixels
    overlap_w = int(overlap * cw)
    overlap_h = int(overlap * ch)
    
    step_h, step_w = ch - overlap_h, cw - overlap_w

    lr_start_w = range(0, w-cw, step_w)  # left to right
    td_start_h = range(0, h-ch, step_h)  # top to down

    coords = itertools.product(lr_start_w, td_start_h)
    crops = [im[h_:h_+ch, w_:w_+cw, :] for w_, h_ in coords]
    return crops
    
    
    



def torchvision_transform():
    seed = np.random.randint(2147483647)
    random.seed(67280421310721)
    torch.manual_seed(67280421310721)



def ext_script():
    # get env parameters
    env_params = get_env_params()
    sources = env_params['sources']
    OUTPUT_BASE_DIR = env_params['OUTPUT_BASE_DIR']

    # create crops
    for crop_size in CROP_SIZE:
        crop_size = 832
        raise NotImplementedError()
        for dataset, (imdir, _, annot_fn) in sources.items():
            dataset_output_dir = os.path.join(OUTPUT_BASE_DIR, str(crop_size), dataset)
            external_crop(imdir, annot_fn, dataset_output_dir, crop_size)

            # filter images with not seg
            seg_files = glob.glob(os.path.join(dataset_output_dir, "seg", "*"))
            n_deleted = 0
            for im_fn in seg_files:
                im = cv2.imread(im_fn)
                # if no annotation, then delete
                if im.sum() == 0:
                    # remove segmentation crop
                    os.remove(im_fn)
                    # remove original crop
                    im_basename = os.path.basename(im_fn)
                    os.remove(os.path.join(dataset_output_dir, "images", im_basename))
                    n_deleted += 1
            print(f"\tdeleted {n_deleted} crops")





def external_crop(input_dir, annot_json_fn, output_base_dir, crop_size, overlap=0.1):
    script_path = os.environ['SCRIPT_PATH']

    # output dirs
    crop_out_im_dir = os.path.join(output_base_dir, "images")
    crop_out_seg_dir = os.path.join(output_base_dir, "seg")

    # create directories if necessary
    os.makedirs(crop_out_im_dir, exist_ok=True)
    os.makedirs(crop_out_seg_dir, exist_ok=True)

    cmd = f"python {script_path} {input_dir} {annot_json_fn} {crop_out_im_dir} {crop_out_seg_dir} {crop_size} {crop_size} {overlap}"
    print(cmd)
    p = sp.Popen(cmd.split(" "))
    p.wait()



def masks_for_unlisted_images(im_dir, seg_dir):
    """
    This function checks if there are any images that don't have a corresponding mask in the mask (seg) directory;
    if there are, an empty mask is created with the same name in the seg_dir
    """
    # read all images and masks
    seg_fns = glob.glob(os.path.join(seg_dir, "*"))
    seg_id_fns = {os.path.basename(fn).split('.')[0]: fn for fn in seg_fns}

    im_fns = glob.glob(os.path.join(im_dir, "*"))
    im_id_fns = {os.path.basename(fn).split('.')[0]: fn for fn in im_fns}

    for im_id, im_path in im_id_fns.items():
        if im_id not in seg_id_fns.keys():
            
            # write empty mask
            im = cv2.imread(im_path)

            # make the image all zeros = empty mask
            im[:,:,:] = 0

            # write mask
            seg_path = os.path.join(seg_dir, f"{im_id}.png")
            cv2.imwrite(seg_path, im)

            print(f"creating mask for {seg_path}")


if __name__ == '__main__':
    #ext_script()
    action = 'crop'
    if action == 'crop':
        main(filter_empty=True)
    elif action == 'create_missing_masks':
        env_params = get_env_params()
        sources = env_params['sources']
        OUTPUT_BASE_DIR = env_params['OUTPUT_BASE_DIR']

        for _, (im_dir, seg_dir) in sources.items():
            masks_for_unlisted_images(im_dir, seg_dir)
