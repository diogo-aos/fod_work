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

RPI_IM_DIR = os.environ['RPI_IMAGE_DIR']
RPI_SEG_DIR = os.environ['RPI_SEG_DIR'] 
RPI_ANNOT_FN = os.environ['RPI_ANNOT_FN']

TX2_IM_DIR = os.environ['TX2_IMAGE_DIR']
TX2_SEG_DIR = os.environ['TX2_SEG_DIR']
TX2_ANNOT_FN = os.environ['TX2_ANNOT_FN']

OUTPUT_BASE_DIR = os.environ['OUTPUT_DIR']



#%%
sources = {
    'rpi': (RPI_IM_DIR, RPI_SEG_DIR, RPI_ANNOT_FN),
    'tx2': (TX2_IM_DIR, TX2_SEG_DIR, TX2_ANNOT_FN)
}

CROP_SIZE = [256, 416, 512, 704, 832, 960]



def main():
    params = itertools.product(CROP_SIZE, sources.keys())
    for crop_size, dataset in params:
        print(crop_size, dataset)
        im_dir, seg_dir, _ = sources[dataset]

        crop_dataset(im_dir, seg_dir,
                    out_dir=os.path.join(OUTPUT_BASE_DIR, str(crop_size)),
                    crop_size=crop_size, overlap=0.5, filter_empty=True)

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

        # get crops
        im_crops = crops = crop_image(im_im, crop_size=256, overlap=overlap)
        seg_crops = crops = crop_image(seg_im, crop_size=256, overlap=overlap)

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
    random.seed(seed)
    torch.manual_seed(seed)



def ext_script():
    # create crops
    for crop_size in CROP_SIZE:
        crop_size = 832
        for dataset, (imdir, _, annot_fn) in sources.items():
            dataset_output_dir = os.path.join(OUTPUT_BASE_DIR, str(crop_size), dataset)
            ext_crop(imdir, annot_fn, dataset_output_dir, crop_size)

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





def ext_crop(input_dir, annot_json_fn, output_base_dir, crop_size, overlap=0.1):
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

if __name__ == '__main__':
    #ext_script()
    main()