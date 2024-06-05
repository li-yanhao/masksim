import os
import glob
from tqdm import tqdm
from typing import List
import argparse

from src.masksim import MaskSim
from src.dataset import MaskSimDataset, balanced_concat_dataloader
import numpy as np
import torch
from torch import utils
from torch.utils.data import (
    SubsetRandomSampler,
    ConcatDataset
)

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


DATASETS = {
    "synthbuster": {
        "sd1": "stable-diffusion-1-3",
        "sd2": "stable-diffusion-2",
        "sdxl": "stable-diffusion-xl",
        "dalle2": "dalle2",
        "dalle3": "dalle3",
        "midjourney": "midjourney-v5",
        "firefly": "firefly",
    },
    "newsynth": {
        "sd1":          "sd1",
        "sd2":          "sd2",
        "sdxl":         "sdxl",
        "dalle2":       "dalle2",
        "dalle3":       "dalle3",
        "midjourney":   "midjourney",
        "firefly":      "firefly",
    }
}

TRAINING_SET = "newsynth"

TRAIN_CLASS_NAMES = [
    "sd1",
    "sd2",
    "sdxl",
    "dalle2",
    "dalle3",
    "midjourney",
    "firefly",
]

IMG_SIZE = 512
CHANNELS = 3
LEARNING_RATE = 1e-3
BATCH_SZ = 8
NUM_WORKERS = 8  # os.cpu_count()  # 6
MAX_EPOCHS = 50

PREPROC = "DnCNN"

def print_info(*text):
    print("\033[1;32m" + " ".join(map(str, text)) + "\033[0;0m")


def train_on_one_class(train_class_name:str, Q:str):
    """ Train detection model for one specific class of synthetic images.

    Parameters
    ----------
    train_class_name : str
        the class name of synthetic images
    Q : int | str
        compression quality factor, "random" for random quality between 70 and uncompressed
    """

    if COMPRESSION_TYPE == "webp":
        ckpt_tag = f"WEBP_Q{Q}"
    elif COMPRESSION_TYPE == "jpeg":
        ckpt_tag = f"JPEG_Q{Q}"
    else:
        ckpt_tag = "uncompressed"

    # ckpt_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    ckpt_tag += f"_w{IMG_SIZE}"

    color_space = "RGB" if Q is None else "YCbCr"
    # compress_type = None if Q is None else "jpeg"

    compress_type = COMPRESSION_TYPE.lower()
    if compress_type is not None:
        assert compress_type in ["jpeg", "webp"]
    else:
        if Q is not None:
            print_info(f"WARNING: compression type is None but the compression quality at {Q} is indicated")

    if Q is None:
        compress_q = None
    elif Q.isdigit():
        compress_q = int(Q)
    else:
        compress_q = None

    img_dir_real_list = [f"processed_data/train/mit5k",
                         f"processed_data/train/coco_train",
                         f"processed_data/train/dresden",
                         f"/gpfs/workdir/liy/datasets/train2014"
                         ]

    real_valid_folder_list = [f"processed_data/train/hdrburst",
                              f"processed_data/train/coco_val"]

    if train_class_name == "ldm":
        fake_train_valid_folder = f"processed_data/train/latent_diffusion"
    else:
        fake_train_valid_folder = f"data/{TRAINING_SET}/{DATASETS[TRAINING_SET][train_class_name]}"

    model = MaskSim(img_size=IMG_SIZE, channels=CHANNELS, lr=LEARNING_RATE,
                    num_masks=1, #num_masks=4,
                    preproc_freeze=False, preprocess=PREPROC).float()
    real_train_dataset = MaskSimDataset(img_dir_real_list=img_dir_real_list, 
                                        img_dir_fake_list=[],
                                        img_size=IMG_SIZE,
                                        channels=CHANNELS,
                                        color_space=color_space,
                                        mode="train",
                                        compress_aug=compress_type,
                                        compress_q=compress_q,
                                        # limit_nb_img=5000
                                        )
    print_info(f"{img_dir_real_list}: {len(real_train_dataset)} images")
    
    real_valid_dataset = MaskSimDataset(img_dir_real_list=real_valid_folder_list, 
                                        img_dir_fake_list=[],
                                        img_size=IMG_SIZE,
                                        channels=CHANNELS,
                                        color_space=color_space,
                                        mode="valid",
                                        compress_aug=compress_type,
                                        compress_q=compress_q
                                        )
    print_info(f"{real_valid_folder_list}: {len(real_valid_dataset)} images")

    fake_dataset = MaskSimDataset(img_dir_real_list=[], 
                                  img_dir_fake_list=[fake_train_valid_folder],
                                  img_size=IMG_SIZE,
                                  channels=CHANNELS,
                                  color_space=color_space,
                                  mode="train",
                                  compress_aug=compress_type,
                                  compress_q=compress_q
                                  )
    print_info(f"{fake_train_valid_folder}: {len(fake_dataset)} images")

    fake_train_dataset, fake_valid_dataset = torch.utils.data.random_split(fake_dataset, [0.9, 0.1])

    # this dataloader is not balanced
    # train_dataset = ConcatDataset([real_train_dataset, fake_train_dataset])
    # train_dataloader = utils.data.DataLoader(train_dataset, batch_size=BATCH_SZ,
    #                                    shuffle=True, num_workers=NUM_WORKERS,
    #                                    collate_fn=MaskSimDataset.collate_fn)

    train_dataloader = balanced_concat_dataloader(
        real_train_dataset, fake_train_dataset, 
        num_workers=NUM_WORKERS, batch_size=BATCH_SZ, collate_fn=MaskSimDataset.collate_fn)

    # not balanced validation dataset is ok
    valid_dataset = ConcatDataset([real_valid_dataset, fake_valid_dataset])
    valid_dataloader = utils.data.DataLoader(valid_dataset, batch_size=BATCH_SZ,
                                       shuffle=False, num_workers=NUM_WORKERS,
                                       collate_fn=MaskSimDataset.collate_fn)

    if VERSION != "":
        version_tag = "_" + VERSION
    else:
        version_tag = ""

    ckpt_path=f"checkpoints{version_tag}/{ckpt_tag}/{TRAINING_SET}"
    os.makedirs(ckpt_path, exist_ok=True)

    high_auc_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_auroc",
        mode="max",
        filename=train_class_name+"-{epoch:02d}-{valid_auroc:.2f}-{valid_loss:.3f}-best_auc",
        dirpath=ckpt_path,
    )

    low_loss_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        filename=train_class_name+"-{epoch:02d}-{valid_auroc:.2f}-{valid_loss:.3f}-low_loss",
        dirpath=ckpt_path,
    )

    last_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        filename="latest-"+train_class_name+"-{epoch:02d}-{valid_acc:.2f}",
        dirpath=ckpt_path,
    )

    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, 
                        accelerator="cpu",
                        # accelerator="gpu", devices=[0],
                        num_nodes=1, deterministic=False,
                        limit_train_batches=1.0,
                        limit_val_batches=1.0,
                        log_every_n_steps=1, profiler=None,
                        enable_checkpointing=True,
                        callbacks=[low_loss_callback],
                        enable_progress_bar=PROGRESS_BAR
                        )

    trainer.fit(model=model, train_dataloaders=train_dataloader,
                val_dataloaders=[valid_dataloader])

    print_info(f"Checkpoint is saved in:", low_loss_callback.best_model_path)


def train_on_multi_classes(train_class_names:List[str], Q:str):
    """ Train detection model for one specific class of synthetic images.

    Parameters
    ----------
    train_class_names : List[str]
        the class name of synthetic images
    Q : int | str
        compression quality factor, "random" for random quality between 70 and uncompressed
    """

    if COMPRESSION_TYPE == "webp":
        ckpt_tag = f"WEBP_Q{Q}"
    elif COMPRESSION_TYPE == "jpeg":
        ckpt_tag = f"JPEG_Q{Q}"
    else:
        ckpt_tag = "uncompressed"

    # ckpt_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    ckpt_tag += f"_w{IMG_SIZE}"

    color_space = "RGB" if Q is None else "YCbCr"

    compress_type = COMPRESSION_TYPE.lower()
    if compress_type is not None:
        assert compress_type in ["jpeg", "webp"]
    else:
        if Q is not None:
            print_info(f"WARNING: compression type is None but the compression quality at {Q} is indicated")

    if Q is None:
        compress_q = None
    elif Q.isdigit():
        compress_q = int(Q)
    else:
        compress_q = None

    # real_train_folder = f"processed_data/uncompressed/raise-2k"
    # real_valid_folder = f"processed_data/uncompressed/hdrburst"
    # fake_train_valid_folder = f"processed_data/{ckpt_tag}/{TRAINING_SET}/{DATASETS[TRAINING_SET][train_class_name]}"
    
    # the right version
    # real_train_folder = f"processed_data/train/raise-2k"
    # real_valid_folder = f"processed_data/train/hdrburst"

    # DEBUG: just for try
    img_dir_real_list = [f"processed_data/train/mit5k",
                         f"processed_data/train/coco_train",
                         f"processed_data/train/dresden",
                         f"/gpfs/workdir/liy/datasets/train2014"
                         ]

    real_valid_folder_list = [f"processed_data/train/hdrburst",
                              f"processed_data/train/coco_val"]

    fake_train_folder_list = []
    for train_class_name in train_class_names:
        if train_class_name == "ldm":
            fake_train_folder = f"processed_data/train/latent_diffusion"
        else:
            fake_train_folder = f"data/{TRAINING_SET}/{DATASETS[TRAINING_SET][train_class_name]}"
        fake_train_folder_list.append(fake_train_folder)

    if train_class_name == "ldm":
        fake_train_valid_folder = f"processed_data/train/latent_diffusion"
    else:
        fake_train_valid_folder = f"data/{TRAINING_SET}/{DATASETS[TRAINING_SET][train_class_name]}"

    model = MaskSim(img_size=IMG_SIZE, channels=CHANNELS, lr=LEARNING_RATE,
                    num_masks=1, #num_masks=4,
                    preproc_freeze=False, preprocess=PREPROC).float()
    real_train_dataset = MaskSimDataset(img_dir_real_list=img_dir_real_list, 
                                        img_dir_fake_list=[],
                                        img_size=IMG_SIZE,
                                        channels=CHANNELS,
                                        color_space=color_space,
                                        mode="train",
                                        compress_aug=compress_type,
                                        compress_q=compress_q,
                                        # limit_nb_img=5000
                                        )
    print_info(f"{img_dir_real_list}: {len(real_train_dataset)} images")
    
    real_valid_dataset = MaskSimDataset(img_dir_real_list=real_valid_folder_list, 
                                        img_dir_fake_list=[],
                                        img_size=IMG_SIZE,
                                        channels=CHANNELS,
                                        color_space=color_space,
                                        mode="valid",
                                        compress_aug=compress_type,
                                        compress_q=compress_q
                                        )
    print_info(f"{real_valid_folder_list}: {len(real_valid_dataset)} images")

    fake_dataset = MaskSimDataset(img_dir_real_list=[], 
                                  img_dir_fake_list=fake_train_folder_list,
                                  img_size=IMG_SIZE,
                                  channels=CHANNELS,
                                  color_space=color_space,
                                  mode="train",
                                  compress_aug=compress_type,
                                  compress_q=compress_q
                                  )
    print_info(f"{fake_train_folder_list}: {len(fake_dataset)} images")

    fake_train_dataset, fake_valid_dataset = torch.utils.data.random_split(fake_dataset, [0.9, 0.1])

    train_dataloader = balanced_concat_dataloader(
        real_train_dataset, fake_train_dataset, 
        num_workers=NUM_WORKERS, batch_size=BATCH_SZ, collate_fn=MaskSimDataset.collate_fn)

    # not balanced validation dataset is ok
    valid_dataset = ConcatDataset([real_valid_dataset, fake_valid_dataset])
    valid_dataloader = utils.data.DataLoader(valid_dataset, batch_size=BATCH_SZ,
                                       shuffle=False, num_workers=NUM_WORKERS,
                                       collate_fn=MaskSimDataset.collate_fn)

    if VERSION != "":
        version_tag = "_" + VERSION
    else:
        version_tag = ""

    ckpt_path=f"checkpoints{version_tag}/{ckpt_tag}/{TRAINING_SET}"
    os.makedirs(ckpt_path, exist_ok=True)

    low_loss_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        filename=train_class_name+"-{epoch:02d}-{valid_auroc:.2f}-{valid_loss:.3f}-low_loss",
        dirpath=ckpt_path,
    )


    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu",
                        devices=[0], #devices=[0,],
                        num_nodes=1, deterministic=False,
                        limit_train_batches=0.1,
                        limit_val_batches=1.0,
                        log_every_n_steps=1, profiler=None,
                        enable_checkpointing=True,
                        callbacks=[low_loss_callback],
                        enable_progress_bar=PROGRESS_BAR
                        )

    trainer.fit(model=model, train_dataloaders=train_dataloader,
                val_dataloaders=[valid_dataloader])

    print_info(f"Checkpoint is saved in:", low_loss_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MaskSim: Detection of synthetic images by masked spectrum similarity analysis. (c) 2024 Yanhao Li. Under license GNU AGPL.'
    )

    parser.add_argument('-Q', '--Q', type=str, required=False,
                        help='compression quality factor', default=None)
    parser.add_argument('-w', '--w', type=int, required=False,
                        help='image size', default=512)
    parser.add_argument('-b', '--batch_sz', type=int, required=False,
                        help='batch size', default=8)
    parser.add_argument('-e', '--epoch', type=int, required=False,
                        help='batch size', default=50)
    parser.add_argument('-p', '--preproc', type=str, required=False,
                        help='preprocessing type', default="DnCNN")
    parser.add_argument('-v', '--version', type=str, required=False,
                        help='version of checkpoints', default="")
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--multiclass',
                        help='training on multiple classes', action='store_true')
    parser.add_argument('--compression', type=str, required=False,
                        help='compression type: jpeg or webp', default=None)


    args = parser.parse_args()
    print(args)

    Q = args.Q
    IMG_SIZE = args.w
    BATCH_SZ = args.batch_sz
    MAX_EPOCHS = args.epoch
    PREPROC = args.preproc
    VERSION = args.version
    PROGRESS_BAR = args.progress
    COMPRESSION_TYPE = args.compression


    if args.multiclass:
        train_on_multi_classes(TRAIN_CLASS_NAMES, Q=Q)
    else:
        for train_class_name in TRAIN_CLASS_NAMES:
            train_on_one_class(train_class_name, Q=Q)
