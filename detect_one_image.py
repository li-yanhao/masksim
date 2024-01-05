import os
import glob
import numpy as np
import torch


from src.masksim import MaskSim
from src.dataset import MaskSimDataset

CLASS_NAMES = [
    "sd1",
    "sd2",
    "sdxl",
    "dalle2",
    "dalle3",
    "midjourney",
    "firefly",
]
IMG_SIZE = 512

def detect_one_image(fname):
    # TODO: as variable
    comp_or_uncomp = "uncompress"

    device = torch.device("cpu")

    # 1. load model
    train_models = dict()
    for class_name in CLASS_NAMES:
        ckpt_fpattern = f"checkpoints/{comp_or_uncomp}/{class_name}" + "*.ckpt"
        ckpt_fnames = glob.glob(ckpt_fpattern)
        ckpt_fnames.sort(key=os.path.getctime)
        ckpt_fname = ckpt_fnames[-1]

        model = MaskSim.load_from_checkpoint(ckpt_fname, map_location=device).float()
        print("Loaded model from ", ckpt_fname)
        model.eval()
        train_models[class_name] = model

    # 2. process image
    dataset = MaskSimDataset(img_dir_real_list=[], img_dir_fake_list=[], img_size=IMG_SIZE,
                             channels=3, color_space="RGB", mode="test")

    imgs = dataset.process(fname, label=None)
    
    # 3. compute scores
    scores_all = []
    with torch.no_grad():
        for class_name in CLASS_NAMES:
            model: MaskSim = train_models[class_name]
            scores : torch.Tensor = model.compute_probs(imgs)
            # real_to_fake_scores.append(scores.detach().cpu().numpy())
            scores_all.append(scores.detach().cpu().numpy())
    
    print("scores_all:", scores_all)
    pass


def inspect(score: float):
    pass


if __name__ == "__main__":
    fname = "fake_001.png"
    detect_one_image(fname)