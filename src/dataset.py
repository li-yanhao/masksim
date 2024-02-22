# Copyright (c) 2023 Yanhao Li
# yanhao.li@outlook.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import glob
import os
from io import BytesIO
from typing import List, Sequence, Iterator, Tuple

import numpy as np
import skimage
from skimage.util import view_as_blocks
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler
import torchvision.transforms as transforms
from PIL import Image

import albumentations as A


def augment_jpeg_pil(img_pil:Image, p:float=1.0, quality=None):
    assert p <= 1.0
    if np.random.rand() < p:
        jpeg_bytes = BytesIO()
        if quality is None:
            quality = np.random.randint(65, 101)
        img_pil.save(jpeg_bytes, format="JPEG", quality=int(quality))
        img_pil = Image.open(jpeg_bytes)
    return img_pil


def augment_webp_pil(img_pil:Image, p:float=1.0) -> Image:
    assert p <= 1.0
    if np.random.rand() < p:
        bytes = BytesIO()
        quality = np.random.randint(75, 101)
        img_pil.save(bytes, format="WebP", quality=int(quality))
        img_pil = Image.open(bytes)
    return img_pil


def augment_resample(imgs: np.ndarray, min_r:float, max_r:float, p:float=1.0, seed:int=None) -> Tuple[np.ndarray, float]:
    # TODO: try different resampling methods
    assert p <= 1.0
    if np.random.rand() < p:
        if seed is not None:
            np.random.seed(seed)
        r = np.random.uniform(min_r, max_r)
        # print("r:", r)

        h_o, w_o = int(imgs.shape[1] * r), int(imgs.shape[2] * r)
        # print("imgs before resizing:", imgs.shape)

        # either using torch interpolation        
        # imgs = torch.from_numpy(imgs)
        # imgs = torch.permute(imgs, (0,3,1,2))
        # imgs: torch.Tensor = torch.nn.functional.interpolate(imgs, size=(h_o, w_o), mode="bilinear")
        # imgs = torch.permute(imgs, (0,2,3,1))
        # imgs = imgs.numpy()
        
        # or using Albumentation
        imgs_o = []
        for idx in range(len(imgs)):
            imgs_o.append(A.augmentations.geometric.resize.Resize(h_o, w_o, interpolation=1, always_apply=True, p=1)(image=imgs[idx])["image"])
        imgs = np.stack(imgs_o)

        # print("imgs after resizing:", imgs.shape)
    else:
        r = 1.0

    # option: Albumentation
    ## A.augmentations.geometric.resize.Resize(height, width, interpolation=1, always_apply=False, p=1)
    # option: skimage
    ## img_o = skimage.transform.rescale(img, r, )
    # option: opencv
    # option: PILLOW
    # option: different interpolations
    
    return imgs, r

def random_crop(img:np.ndarray, crop_size:int) -> np.ndarray:
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
    x1, y1 = x1 // 8 * 8, y1 // 8 * 8
    x2, y2 = x1 + crop_size, y1 + crop_size
    return img[y1:y2, x1:x2]


def divide_in_patches(img:np.ndarray, patch_size:int) -> np.ndarray:
    H, W, C = img.shape
    img = img[:H // patch_size * patch_size, :W // patch_size * patch_size, :]
    patches = view_as_blocks(img, (patch_size, patch_size, C)) # nh, nw, 1, sz, sz, C
    patches = patches.reshape(-1, patch_size, patch_size, C)
    return patches


class WeightedSubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], weights: Sequence[float], generator=None) -> None:
        self.indices = indices
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for index in np.random.choice(self.indices, size=len(self.indices), p=self.weights):
            yield index

    def __len__(self) -> int:
        return len(self.indices)
    

class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.drop_last = drop_last
        self.batch_size = batch_size
        
        self.rng = np.random.default_rng(12345)
        self.rd_seed = 0

    def __iter__(self):
        batch = []
        
        for idx in self.sampler:
            batch.append((idx, self.rd_seed))
            if len(batch) == self.batch_size:
                yield batch

                self.rd_seed = self.rng.integers(2**20)
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch
        self.rd_seed = self.rng.integers(2**20)

    
class MaskSimDataset(Dataset):
    def __init__(self, img_dir_fake_list:List[str], 
                 img_dir_real_list:List[str],
                 img_size:int,
                 channels:int=3,
                 color_space:str="YCbCr",
                 compress_aug:str=None,
                 fix_q:int=None,
                 rescale_min:float=1.0,
                 rescale_max:float=1.0,
                 mode:str="valid",
                 limit_nb_img:int=None):
        super().__init__()
        self.img_fnames = []
        self.labels = []
        self.channels = channels
        self.mode = mode
        self.img_size = img_size
        self.color_space = color_space

        self.rescale_min = rescale_min
        self.rescale_max = rescale_max
        
        if compress_aug is not None:
            assert compress_aug in ["jpeg", "webp"]
        self.compress_aug = compress_aug
        self.fix_q = fix_q

        self.fnames_real = []
        for img_dir in img_dir_real_list:
            for ext in ["tiff", "png", "jpg", "webp"]:
                self.fnames_real.extend(
                    glob.glob(
                        os.path.join(img_dir, "**/*.{}".format(ext)), recursive=True
                    )
                )
        self.fnames_real.sort()

        self.fnames_fake = []
        for img_dir in img_dir_fake_list:
            for ext in ["tiff", "png", "jpg", "webp"]:
                self.fnames_fake.extend(
                    glob.glob(
                        os.path.join(img_dir, "**/*.{}".format(ext)), recursive=True
                    )
                )
        self.fnames_fake.sort()

        self.fnames = self.fnames_real + self.fnames_fake
        self.labels.extend([0] * len(self.fnames_real))
        self.labels.extend([1] * len(self.fnames_fake))

        # if self.mode != "test":
            # shuffle in advance the positives and negatives
        
        self.fnames = np.array(self.fnames)
        self.labels = np.array(self.labels)
        indices = np.arange(len(self.fnames))
        np.random.shuffle(indices)
        self.fnames = self.fnames[indices]
        self.labels = self.labels[indices]

        if limit_nb_img is not None:
            self.fnames = self.fnames[:limit_nb_img]
            self.labels = self.labels[:limit_nb_img]

    def process(self, fname:str, label:int, rd_seed=int) -> Tuple[torch.Tensor, float]:
        img = skimage.io.imread(fname)
        if np.ndim(img) == 2 or img.shape[0] < self.img_size or img.shape[1] < self.img_size:
            return None

        img = img[:, :, :3]
        
        if self.mode == "train":
            img = random_crop(img, self.img_size)
            imgs = img[None, ...]
        else:
            img = center_crop(img, self.img_size)
            imgs = img[None, ...]

        if self.mode == "train" and label == 0 and np.random.rand() < 0.9:
            # only augment real samples during training
            for idx in range(len(imgs)):
                imgs[idx] = A.augmentations.transforms.GaussNoise(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.transforms.Sharpen(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.Defocus(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.transforms.CLAHE(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.transforms.RGBShift(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.transforms.ColorJitter(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.transforms.HueSaturationValue(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.transforms.RandomGamma(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.transforms.Emboss(p=0.5)(image=imgs[idx])["image"]
                imgs[idx] = A.augmentations.geometric.rotate.Rotate(p=0.5)(image=imgs[idx])["image"]

        # pipeline: crop -> resample -> jpeg 
        imgs, scale = augment_resample(imgs, self.rescale_min, self.rescale_max, p=1.0, seed=rd_seed)

        # TODO: it doesn't work so far
        # if (self.compress_aug is not None) and (self.mode in ["train"]) and (label == 0):
        #     for idx in range(len(imgs)):
        #         img_pil = Image.fromarray(imgs[idx])
        #         if self.compress_aug == "jpeg":
        #             img_pil = augment_jpeg_pil(img_pil, p=1.0, quality=self.fix_q)
        #         elif self.compress_aug == "webp":
        #             img_pil = augment_webp_pil(img_pil, p=1.0)
        #         imgs[idx] = np.array(img_pil)
        
        # resample

        if (self.compress_aug is not None) and (self.mode in ["train", "valid"]):
            for idx in range(len(imgs)):
                img_pil = Image.fromarray(imgs[idx])
                if self.compress_aug == "jpeg":
                    img_pil = augment_jpeg_pil(img_pil, p=1.0, quality=self.fix_q)
                elif self.compress_aug == "webp":
                    img_pil = augment_webp_pil(img_pil, p=1.0)
                imgs[idx] = np.array(img_pil)


        if self.color_space == "YCbCr":
            for idx in range(len(imgs)):
                imgs[idx] = skimage.color.rgb2ycbcr(imgs[idx])
        imgs = imgs / 255.0

        imgs = np.transpose(imgs, (0, 3, 1, 2))
        imgs = torch.from_numpy(imgs).float() # N, H, W, C

        return imgs, scale

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, item):
        index, rd_seed = item
        fname = self.fnames[index]
        label = self.labels[index]
        imgs, scale = self.process(fname, label, rd_seed)

        while imgs is None:
            index = np.random.randint(0, len(self.fnames))
            fname = self.fnames[index]
            label = self.labels[index]
            imgs, scale = self.process(fname, label, rd_seed)
        return imgs, scale, label

    def get_data_sampler(self) -> WeightedRandomSampler:
        labels = np.array(self.labels)
        samples_weight = torch.ones(len(labels))
        samples_weight[ labels == 0 ] = 1 / np.sum(labels == 0 )
        samples_weight[ labels == 1 ] = 1 / np.sum(labels == 1 )
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler
    
    @staticmethod
    def collate_fn(data):
        imgs = [ d[0] for d in data ] # list of Ni x C x H x W
        scales = torch.as_tensor([ d[1] for d in data ])
        # print("scales in collate_fn(): ", scales)

        imgs = torch.concat(imgs)
        labels = [ torch.ones(d[0].shape[0]) * d[2] for d in data] # list of scalars
        labels = torch.concat(labels)

        max_batch_sz = 64
        if len(imgs) > max_batch_sz:
            indices = torch.randperm(len(imgs))
            imgs = imgs[indices[:max_batch_sz]]
            labels = labels[indices[:max_batch_sz]]

        return imgs, scales, labels
