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
from typing import List, Sequence, Iterator

import numpy as np
import skimage
from skimage.util import view_as_blocks
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler, ConcatDataset, DataLoader
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
        quality = np.random.randint(65, 101)
        img_pil.save(bytes, format="WebP", quality=int(quality))
        img_pil = Image.open(bytes)
    return img_pil


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


def float_to_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


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


def balanced_concat_dataloader(ds_0: Dataset, ds_1: Dataset,
                               num_workers:int=0,
                               batch_size:int=1,
                               collate_fn=None) -> DataLoader:
    """ Concat two datasets and return a dataloader with balanced sampler """
    dataset = ConcatDataset((ds_0, ds_1))
    samples_weight = torch.cat((torch.ones(len(ds_0)) / len(ds_0),
                                torch.ones(len(ds_1)) / len(ds_1)))
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
        collate_fn=collate_fn)

    return loader


class MaskSimDataset(Dataset):
    def __init__(self, img_dir_fake_list:List[str], 
                 img_dir_real_list:List[str],
                 img_size:int,
                 channels:int=3,
                 color_space:str="YCbCr",
                 compress_aug:str=None,
                 compress_q:int=None,  # if None, random quality is used
                 mode:str="valid",
                 limit_nb_img:int=None,
                 return_fname=False):
        super().__init__()

        self.img_fnames = []
        self.labels = []
        self.channels = channels
        self.mode = mode
        self.img_size = img_size
        self.color_space = color_space
        self.return_fname = return_fname
        
        if compress_aug is not None:
            assert compress_aug in ["jpeg", "webp"]
        self.compress_aug = compress_aug
        self.compress_q = compress_q

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

        # shuffle in advance the positives and negatives
        if self.mode != "test":
            self.fnames = np.array(self.fnames)
            self.labels = np.array(self.labels)
            indices = np.arange(len(self.fnames))
            np.random.shuffle(indices)
            self.fnames = self.fnames[indices]
            self.labels = self.labels[indices]

        if limit_nb_img is not None:
            self.fnames = self.fnames[:limit_nb_img]
            self.labels = self.labels[:limit_nb_img]

    def process(self, fname:str, label:int):
        img = skimage.io.imread(fname)
        if np.ndim(img) == 2 or img.shape[0] < self.img_size or img.shape[1] < self.img_size:
            print(f"ERROR: the input size must be larger than {self.img_size} both in width and length.")
            return None

        img = img[:, :, :3]
        if self.mode == "train":
            # resize pristine images
            if label == 0:
                if np.random.rand() < 0.5:
                    up_ratio = self.img_size / min(img.shape[0], img.shape[1]) * np.random.uniform(1, 1.5)
                    final_h, final_w = int(up_ratio * img.shape[0]), int(up_ratio * img.shape[1])
                    img = skimage.transform.resize(img, (final_h, final_w), preserve_range=True, anti_aliasing=False)
                    img = float_to_uint8(img)
                elif min(img.shape[0], img.shape[1]) < self.img_size:
                    up_ratio = self.img_size / (min(img.shape[0], img.shape[1]) + 1)
                    final_h, final_w = int(up_ratio * img.shape[0]), int(up_ratio * img.shape[1])
                    img = skimage.transform.resize(img, (final_h, final_w), preserve_range=True, anti_aliasing=False)
                    img = float_to_uint8(img)

            img = random_crop(img, self.img_size)
            imgs = img[None, ...]
        else:
            img = center_crop(img, self.img_size)
            imgs = img[None, ...]

        if (self.compress_aug is not None) and (self.mode in ["train", "valid"]):
            for idx in range(len(imgs)):
                img_pil = Image.fromarray(imgs[idx])
                if self.compress_aug == "jpeg":
                    if self.compress_q is not None: 
                        img_pil = augment_jpeg_pil(img_pil, p=1.0, quality=self.compress_q)
                    else:
                        img_pil = augment_jpeg_pil(img_pil, p=0.95)
                elif self.compress_aug == "webp":
                    img_pil = augment_webp_pil(img_pil, p=1.0)
                imgs[idx] = np.array(img_pil)


        if self.color_space == "YCbCr":
            for idx in range(len(imgs)):
                imgs[idx] = skimage.color.rgb2ycbcr(imgs[idx])
        imgs = imgs / 255.0

        imgs = np.transpose(imgs, (0, 3, 1, 2))
        imgs = torch.from_numpy(imgs).float() # N, H, W, C
        return imgs

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index):
        fname = self.fnames[index]
        label = self.labels[index]
        imgs = self.process(fname, label)

        while imgs is None:
            index = np.random.randint(0, len(self.fnames))
            fname = self.fnames[index]
            label = self.labels[index]
            imgs = self.process(fname, label)

        if self.return_fname:
            return imgs, label, fname
        else:
            return imgs, label

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
        imgs = torch.concat(imgs)
        labels = [ torch.ones(d[0].shape[0]) * d[1] for d in data] # list of scalars
        labels = torch.concat(labels)

        if len(data[0]) == 3:
            return_fname = True
            fnames = [d[2] for d in data]
        else:
            return_fname = False

        max_batch_sz = 64
        if len(imgs) > max_batch_sz:
            indices = torch.randperm(len(imgs))
            imgs = imgs[indices[:max_batch_sz]]
            labels = labels[indices[:max_batch_sz]]
        
        if return_fname:
            return imgs, labels, fnames
        else:
            return imgs, labels
