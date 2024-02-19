import glob
import os

import numpy as np
import skimage
from skimage.util import view_as_blocks
from PIL import Image

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

NEWSYNTH_UNCOMP_LIST = [
    "data/newsynth/stable-diffusion-1",
    "data/newsynth/stable-diffusion-2",
    "data/newsynth/stable-diffusion-xl",
    "data/newsynth/dalle2",
    "data/newsynth/dalle3",
    "data/newsynth/midjourney",
    "data/newsynth/firefly",
]

REAL_UNCOMP_LIST = [
    # "data/raise-2k",
    "data/hdrburst",
    "data/dresden",
]

IMG_SIZE = 512
NUM_WORKERS = 10


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


def main_fake(Q:int):
    """ Preprocess fake images
    Parameters
    ----------
    Q : int
        Compression quality factor
    """

    for dataset_name, folder_list in zip(["synthbuster", "newsynth"], [SYNTHBUSTER_UNCOMP_LIST, NEWSYNTH_UNCOMP_LIST]):
        for in_folder in folder_list:
            class_name = os.path.basename(in_folder)
            
            if Q is None:
                out_folder = os.path.join(f"processed_data/uncompressed", dataset_name, class_name)
            else:
                out_folder = os.path.join(f"processed_data/JPEG_Q{Q}", dataset_name, class_name)
                # out_folder = os.path.join(f"processed_data/WEBP_Q{Q}", dataset_name, class_name)
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
                # img = skimage.io.imread(fname)
                img_pil = Image.open(fname)
                img = np.array(img_pil)

                if np.ndim(img) == 2:  # gray image
                    img = skimage.color.gray2rgb(img)
                elif img.shape[2] == 4:  # RGBA image
                    img = img[:, :, :3]
                
                img = center_crop(img, IMG_SIZE)
                # img = random_crop(img, IMG_SIZE)
                
                if Q is None:  # uncompressed image
                    save_fname = os.path.basename(fname)
                    save_fname = os.path.join(out_folder, save_fname)

                    img_pil = Image.fromarray(img)
                    img_pil.save(save_fname)
                else:  # compressed image
                    save_fname = os.path.basename(fname)
                    save_fname = os.path.join(out_folder, save_fname.replace(".png", ".jpg"))
                    # save_fname = os.path.join(out_folder, save_fname.replace(".png", ".webp"))

                    img_pil = Image.fromarray(img)
                    img_pil.save(save_fname, format="jpeg", quality=Q)
                    # img_pil.save(save_fname, format="webp", quality=Q)

            
            for fname in fnames:
                pool.submit(task, fname)
            
            while not pool.empty():
                pool.pop_result()


def main_real(Q:int):
    """ Preprocess real images
    Parameters
    ----------
    Q : int
        Compression quality factor
    """

    for in_folder in REAL_UNCOMP_LIST:
        class_name = os.path.basename(in_folder)

        if Q is None:
            out_folder = os.path.join(f"processed_data/uncompressed", class_name)
        else:
            out_folder = os.path.join(f"processed_data/JPEG_Q{q}", class_name)
            # out_folder = os.path.join(f"processed_data/WEBP_Q{q}", class_name)

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
            
            # img = center_crop(img, IMG_SIZE)
            img = random_crop(img, IMG_SIZE)

            if Q is None:  # uncompressed image
                save_fname = os.path.basename(fname)
                save_fname = os.path.join(out_folder, save_fname)

                img_pil = Image.fromarray(img)
                img_pil.save(save_fname)
            else:  # compressed image
                save_fname = os.path.basename(fname)
                save_fname = os.path.join(out_folder, save_fname.replace(".png", ".jpg"))
                # save_fname = os.path.join(out_folder, save_fname.replace(".png", ".webp"))

                img_pil = Image.fromarray(img)
                img_pil.save(save_fname, format="jpeg", quality=Q)
                # img_pil.save(save_fname, format="webp", quality=Q)

        pool = ThreadPoolPlus(workers=NUM_WORKERS)
        for fname in fnames:
            pool.submit(task, fname)
        
        while not pool.empty():
            pool.pop_result()


def main_cut(cut_size:int):
    """ Cut large images into small images such that loading data during training is more efficient. """
    for in_folder in REAL_UNCOMP_LIST:
        class_name = os.path.basename(in_folder)

        out_folder = os.path.join(f"processed_data/train", class_name)

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
            
            imgs = divide_in_patches(img, cut_size)

            for i in range(len(imgs)):
                save_fname = os.path.basename(fname)
                save_fname = os.path.join(out_folder, save_fname)
                save_fname = save_fname[:-4] + f"_{i:03d}" + save_fname[-4:]
                img_pil = Image.fromarray(imgs[i])
                img_pil.save(save_fname)

        pool = ThreadPoolPlus(workers=NUM_WORKERS)
        for fname in fnames:
            pool.submit(task, fname)
        
        while not pool.empty():
            pool.pop_result()


if __name__ == "__main__":

    # preprocess data for training
    main_cut(cut_size=1024)

    # # preprocess data for test
    # for q in [None, 90, 80, 70]:
    #     print(f"compression quality: {q}")
    #     main_real(q)
    #     main_fake(q)
