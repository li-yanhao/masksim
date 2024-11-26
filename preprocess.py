import glob
import os

import skimage
import numpy as np
from PIL import Image
from skimage.util import view_as_blocks

from src.thread_pool_plus import ThreadPoolPlus


SYNTHBUSTER_UNCOMP_LIST = [
    "data/synthbuster/stable-diffusion-1-3",
    "data/synthbuster/stable-diffusion-2",
    "data/synthbuster/stable-diffusion-xl",
    "data/synthbuster/dalle2",
    "data/synthbuster/dalle3",
    "data/synthbuster/midjourney-v5",
    "data/synthbuster/firefly",
]
NUM_WORKERS = 8


def random_crop(img, crop_size:int):
    h_orig, w_orig = img.shape[:2]

    if crop_size <= w_orig:
        x1 = np.random.randint(0,  w_orig - crop_size + 1)
    else:
        x1 = 0
    if crop_size <= h_orig:
        y1 = np.random.randint(0,  h_orig - crop_size + 1)
    else:
        y1 = 0
    return img[y1: (y1+crop_size), x1: (x1+crop_size)]


def center_crop(img:np.ndarray, crop_size:int) -> np.ndarray:
    h_orig, w_orig = img.shape[:2]
    h_center, w_center = h_orig // 2, w_orig // 2
    y1 = h_center - crop_size // 2
    x1 = w_center - crop_size // 2
    y2 = h_center + crop_size // 2
    x2 = w_center + crop_size // 2
    x1, y1 = max(0, x1), max(0, y1)
    return img[y1:y2, x1:x2]


def divide_in_patches(img:np.ndarray, patch_size:int) -> np.ndarray:
    H, W, C = img.shape
    img = img[:H // patch_size * patch_size, :W // patch_size * patch_size, :]
    patches = view_as_blocks(img, (patch_size, patch_size, C)) # nh, nw, 1, sz, sz, C
    patches = patches.reshape(-1, patch_size, patch_size, C)
    return patches


def float_to_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def main_test_fake(crop_size:int, Q:str, compression="jpeg"):
    """ Preprocess fake images for test.

    Parameters
    ----------
    crop_size : int
        Cropping size
    Q : str
        Compression quality factor
    compression : str
        Compression type, "jpeg" or "webp"
    """

    dataset_name = "synthbuster"
    for in_folder in SYNTHBUSTER_UNCOMP_LIST:
        print(f"Processing {in_folder}")
        class_name = os.path.basename(in_folder)
        
        if Q is None:
            out_folder = os.path.join(f"processed_data/uncompressed", dataset_name, class_name)
        else:
            if compression == 'jpeg':
                out_folder = os.path.join(f"processed_data/JPEG_Q{Q}", dataset_name, class_name)
            elif compression == 'webp':
                out_folder = os.path.join(f"processed_data/WEBP_Q{Q}", dataset_name, class_name)
        os.makedirs(out_folder, exist_ok=True)

        fnames = []
        for ext in ["png"]:
            fnames.extend(
                glob.glob(
                    os.path.join(in_folder, "**/*.{}".format(ext)), recursive=True
                )
            )
        fnames.sort()

        pool = ThreadPoolPlus(workers=NUM_WORKERS)
        def task(fname):
            img_pil = Image.open(fname)
            img = np.array(img_pil)

            if np.ndim(img) == 2:  # gray image
                img = skimage.color.gray2rgb(img)
            elif img.shape[2] == 4:  # RGBA image
                img = img[:, :, :3]

            img = center_crop(img, crop_size)
            
            save_fname = None
            if Q is None:  # uncompressed image
                save_fname = os.path.basename(fname)
                save_fname = os.path.join(out_folder, save_fname)

                img_pil = Image.fromarray(img)
                img_pil.save(save_fname)
            else:  # compressed image
                if Q == "random":
                    quality = np.random.randint(65, 101)
                else:
                    quality = int(Q)
                if compression == 'jpeg':
                    save_fname = os.path.basename(fname)
                    save_fname = os.path.join(out_folder, save_fname.replace(".png", ".jpg"))
                    img_pil = Image.fromarray(img)

                    img_pil.save(save_fname, format="jpeg", quality=quality)
                elif compression == 'webp':
                    save_fname = os.path.basename(fname)
                    save_fname = os.path.join(out_folder, save_fname.replace(".png", ".webp"))
                    img_pil = Image.fromarray(img)
                    img_pil.save(save_fname, format="webp", quality=quality)

            return save_fname

        for fname in fnames:
            pool.submit(task, fname)
        
        while not pool.empty():
            pool.pop_result()


def main_test_real(crop_size:int, Q:str, compression="jpeg"):
    """ Preprocess real images for test.
    
    Parameters
    ----------
    crop_size : int
        Cropping size
    Q : str
        Compression quality factor
    compression : str
        Compression type, "jpeg" or "webp"
    """

    in_folder = "data/raise"
    class_name = "raise"

    print(f"Processing {in_folder}")

    if Q is None:
        out_folder = os.path.join(f"processed_data/uncompressed", class_name)
    else:
        if compression == 'jpeg':
            out_folder = os.path.join(f"processed_data/JPEG_Q{Q}", class_name)
        elif compression == 'webp':
            out_folder = os.path.join(f"processed_data/WEBP_Q{Q}", class_name)

    os.makedirs(out_folder, exist_ok=True)

    fnames = []
    for ext in ["png"]:
        fnames.extend(
            glob.glob(
                os.path.join(in_folder, "**/*.{}".format(ext)), recursive=True
            )
        )
    fnames.sort()

    def task(fname):
        img_pil = Image.open(fname)
        img = np.array(img_pil)
        if len(img.shape) == 2:  # gray image
            img = skimage.color.gray2rgb(img)
        elif img.shape[2] == 4:  # RGBA image
            img = img[:, :, :3]
        
        img = center_crop(img, crop_size)

        if (img.shape[0], img.shape[1]) != (crop_size, crop_size):
            print(f"Warning: {fname} is smaller than the cropping size {crop_size}x{crop_size}")
            return None

        save_fname = None
        if Q is None:  # uncompressed image
            save_fname = os.path.basename(fname)
            save_fname = os.path.join(out_folder, save_fname)

            img_pil = Image.fromarray(img)
            img_pil.save(save_fname)
        else:  # compressed image
            if Q == "random":
                quality = np.random.randint(65, 101)
            else:
                quality = int(Q)
            if compression == "jpeg":
                save_fname = os.path.basename(fname)
                save_fname = os.path.join(out_folder, save_fname.replace(".png", ".jpg"))
                img_pil = Image.fromarray(img)
                img_pil.save(save_fname, format="jpeg", quality=quality)
            elif compression == "webp":
                save_fname = os.path.basename(fname)
                save_fname = os.path.join(out_folder, save_fname.replace(".png", ".webp"))
                img_pil = Image.fromarray(img)
                img_pil.save(save_fname, format="webp", quality=quality)
        
        return save_fname

    pool = ThreadPoolPlus(workers=NUM_WORKERS)
    for fname in fnames:
        pool.submit(task, fname)
    
    # save the scales of the images
    while not pool.empty():
        pool.pop_result()


if __name__ == "__main__":
    main_test_real(Q="random", crop_size=512, compression="jpeg")
    main_test_fake(Q="random", crop_size=512, compression="jpeg")
