import argparse
import glob
import os

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

CLASS_FULLNAME_DICT = {
    "sd1": "stable-diffusion-1",
    "sd2": "stable-diffusion-2",
    "sdxl":  "stable-diffusion-xl",
    "glide":  "glide",
    "dalle2":"dalle2",
    "dalle3":"dalle3",
    "midjourney": "midjourney",
    "firefly": "firefly",
}

IMG_SIZE = 512

ROOT = os.path.dirname(os.path.realpath(__file__))


def detect_one_image(fname: str):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 1. load model
    train_models = dict()
    for class_name in CLASS_NAMES:
        ckpt_fpattern = os.path.join(ROOT, f"checkpoints/JPEG_Qrandom_w512/newsynth/", class_name + "*.ckpt")
        ckpt_fnames = glob.glob(ckpt_fpattern)
        ckpt_fnames.sort(key=os.path.getctime)
        ckpt_fname = ckpt_fnames[-1]

        model = MaskSim.load_from_checkpoint(ckpt_fname, map_location=device).float()
        model.eval()
        train_models[class_name] = model.to(device)

    # 2. process image
    dataset = MaskSimDataset(img_dir_real_list=[], img_dir_fake_list=[], img_size=IMG_SIZE,
                             channels=3, color_space="YCbCr", mode="test")

    imgs = dataset.process(fname, label=None)

    if imgs is None:
        exit(0)
    
    # 3. compute scores
    score_final = -1
    class_full_name_final = None
    with torch.no_grad():
        for class_name in CLASS_NAMES:
            model: MaskSim = train_models[class_name]
            imgs = imgs.to(device)
            scores : torch.Tensor = model.compute_probs(imgs)
            score = scores.cpu()[0]
            class_full_name = CLASS_FULLNAME_DICT[class_name]
            print("Score from the detector of {}: {}".format(class_full_name, scores[0]))

            if score > score_final:
                class_full_name_final = class_full_name
                score_final = score

    print()
    print("Maximum score from the detector of {} as the final score: {}".format(class_full_name_final, score_final))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='python detect_one_image.py',
        description='MaskSim: Detection of synthetic images by masked spectrum similarity analysis. (c) 2024 Yanhao Li. Under license GNU AGPL.'
    )

    parser.add_argument('-i', '--img_filename', type=str, required=True,
                        help='tested image file name')
    args = parser.parse_args()

    fname = args.img_filename

    detect_one_image(fname)
