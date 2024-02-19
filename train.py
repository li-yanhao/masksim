import os
import glob
from tqdm import tqdm
from typing import List
import argparse

from src.masksim import MaskSim
from src.dataset import MaskSimDataset
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
        "sd1":          "stable-diffusion-1",
        "sd2":          "stable-diffusion-2",
        "sdxl":         "stable-diffusion-xl",
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
LEARNING_RATE = 1e-4
# LEARNING_RATE = 0.001
# LEARNING_RATE = 5e-5 # DnCNN
# LEARNING_RATE = 1e-3 # no filter
BATCH_SZ = 8
NUM_WORKERS = 8  # 6
MAX_EPOCHS = 50


def print_info(*text):
    print("\033[1;32m" + " ".join(map(str, text)) + "\033[0;0m")


def train_on_one_class(train_class_name:str, Q:int):
    """ Train detection model for one specific class of synthetic images.

    Parameters
    ----------
    train_class_name : str
        the class name of synthetic images
    Q : int
        compression quality factor
    """

    ckpt_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    ckpt_tag += f"_w{IMG_SIZE}"

    color_space = "RGB" if Q is None else "YCbCr"
    compress_type = None if Q is None else "jpeg"
    
    # real_train_folder = f"processed_data/uncompressed/raise-2k"
    # real_valid_folder = f"processed_data/uncompressed/hdrburst"
    # fake_train_valid_folder = f"processed_data/{ckpt_tag}/{TRAINING_SET}/{DATASETS[TRAINING_SET][train_class_name]}"
    
    # the right version
    # real_train_folder = f"processed_data/train/raise-2k"
    # real_valid_folder = f"processed_data/train/hdrburst"

    # DEBUG: just for try
    real_train_folder = f"processed_data/train/dresden"
    real_valid_folder = f"processed_data/train/hdrburst"

    fake_train_valid_folder = f"data/{TRAINING_SET}/{DATASETS[TRAINING_SET][train_class_name]}"

    model = MaskSim(img_size=IMG_SIZE, channels=CHANNELS, lr=LEARNING_RATE,
                    num_masks=1, preproc_freeze=False, preprocess="DnCNN").float()
    real_train_dataset = MaskSimDataset(img_dir_real_list=[real_train_folder], 
                                        img_dir_fake_list=[],
                                        img_size=IMG_SIZE,
                                        channels=CHANNELS,
                                        color_space=color_space,
                                        mode="train",
                                        compress_aug=compress_type,
                                        fix_q=Q, limit_nb_img=2000
                                        )
    print_info(f"{real_train_folder}: {len(real_train_dataset)} images")
    
    real_valid_dataset = MaskSimDataset(img_dir_real_list=[real_valid_folder], 
                                        img_dir_fake_list=[],
                                        img_size=IMG_SIZE,
                                        channels=CHANNELS,
                                        color_space=color_space,
                                        mode="valid",
                                        compress_aug=compress_type,
                                        fix_q=Q
                                        )
    print_info(f"{real_valid_folder}: {len(real_valid_dataset)} images")

    fake_dataset = MaskSimDataset(img_dir_real_list=[], 
                                  img_dir_fake_list=[fake_train_valid_folder],
                                  img_size=IMG_SIZE,
                                  channels=CHANNELS,
                                  color_space=color_space,
                                  mode="train",
                                  compress_aug=compress_type,
                                  fix_q=Q
                                  )
    print_info(f"{fake_train_valid_folder}: {len(fake_dataset)} images")

    fake_train_dataset, fake_valid_dataset = torch.utils.data.random_split(fake_dataset, [0.9, 0.1])
    train_dataset = ConcatDataset([real_train_dataset, fake_train_dataset])
    valid_dataset = ConcatDataset([real_valid_dataset, fake_valid_dataset])
    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=BATCH_SZ,
                                       shuffle=True, num_workers=NUM_WORKERS,
                                       collate_fn=MaskSimDataset.collate_fn)
    valid_dataloader = utils.data.DataLoader(valid_dataset, batch_size=BATCH_SZ,
                                       shuffle=False, num_workers=NUM_WORKERS,
                                       collate_fn=MaskSimDataset.collate_fn)
    
    ckpt_path=f"checkpoints/{ckpt_tag}/{TRAINING_SET}"
    os.makedirs(ckpt_path, exist_ok=True)

    best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_acc",
        mode="max",
        filename=train_class_name+"-{epoch:02d}-{valid_acc:.2f}-{valid_loss:.3f}",
        dirpath=ckpt_path,
    )

    low_loss_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        filename="low_loss-"+train_class_name+"-{epoch:02d}-{valid_loss:.3f}",
        dirpath=ckpt_path,
    )

    last_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        filename="latest-"+train_class_name+"-{epoch:02d}-{valid_acc:.2f}",
        dirpath=ckpt_path,
    )

    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu",
                        devices=[0,], #devices=[0,],
                        num_nodes=1, deterministic=False,
                        limit_train_batches=1.0,
                        limit_val_batches=1.0,
                        log_every_n_steps=1, profiler=None,
                        enable_checkpointing=True,
                        callbacks=[best_callback],
                        enable_progress_bar=False
                        )

    trainer.fit(model=model, train_dataloaders=train_dataloader,
                val_dataloaders=[valid_dataloader])

    print_info(f"Checkpoint is saved in:", best_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MaskSim: Detection of synthetic images by masked spectrum similarity analysis. (c) 2024 Yanhao Li. Under license GNU AGPL.'
    )

    parser.add_argument('-Q', '--Q', type=int, required=False,
                        help='compression quality factor', default=None)
    parser.add_argument('-w', '--w', type=int, required=False,
                        help='image size', default=512)
    parser.add_argument('-b', '--batch_sz', type=int, required=False,
                        help='batch size', default=8)
    args = parser.parse_args()

    Q = args.Q
    IMG_SIZE = args.w
    BATCH_SZ = args.batch_sz

    for train_class_name in TRAIN_CLASS_NAMES:
        train_on_one_class(train_class_name, Q=Q)
