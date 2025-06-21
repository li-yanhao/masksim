import argparse
import glob
import os
from typing import List
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
import plotly.graph_objects as go

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




# SDXL:


# Dalle2:


# Dalle3:


# Midjourney:


# Firefly:

    

# thresholds for FPR=0.01, 0.05, 0.1 for each model
THRESHOLDS_MODELS = {
    "sd1": [
        0.9838761669397355, 
        0.6193085968494393, 
        0.1568259432911874
    ], 
    "sd2": [
        0.9969909167289733,
        0.8133679926395408,
        0.39108284115791325
    ],
    "sdxl": [
        0.9192868810892096,
        0.1516933783888815,
        0.0364518601447344
    ],
    "dalle2": [
        0.9986862266063691,
        0.9710189938545225,
        0.8885011434555055
    ],
    "dalle3": [
        0.9217132574319838,
        0.5467243432998652,
        0.3076250642538071
    ],
    "midjourney": [
        0.9979619538784027,
        0.946184405684471,
        0.7582032501697541
    ],
    "firefly": [
        0.9273038148880005,
        0.5623477935791015,
        0.3056519240140915
    ], 
    "max": [
        0.9998424160480499,
        0.9951299488544465,
        0.9781222820281982
    ]
}

VISUALIZATION_RANGE = {
    "sd1": [1.8e-04, 3.8e-05],
    "sd2": [1.1e-04, 3.6e-05],
    "sdxl": [1.2e-04, 3.6e-05],
    "dalle2": [6.2e-05, 3.9e-05],
    "dalle3": [9.8e-05, 4.8e-05],
    "midjourney": [6.6e-05, 2.6e-05],
    "firefly": [4.8e-04, 1.4e-04],
}

IMG_SIZE = 512

ROOT = os.path.dirname(os.path.realpath(__file__))

NEG_SCORE_FNAME_DICT = {
    "sd1": ["calibration/coco_unlabled_2017_to_sd1.csv",
            "calibration/FlickrFace_to_sd1.csv"],
    "sd2": ["calibration/coco_unlabled_2017_to_sd2.csv",
            "calibration/FlickrFace_to_sd2.csv"],
    "sdxl": ["calibration/coco_unlabled_2017_to_sdxl.csv",
            "calibration/FlickrFace_to_sdxl.csv"],
    "dalle2": ["calibration/coco_unlabled_2017_to_dalle2.csv",
            "calibration/FlickrFace_to_dalle2.csv"],
    "dalle3": ["calibration/coco_unlabled_2017_to_dalle3.csv",
            "calibration/FlickrFace_to_dalle3.csv"],
    "midjourney": ["calibration/coco_unlabled_2017_to_midjourney.csv",
            "calibration/FlickrFace_to_midjourney.csv"],
    "firefly": ["calibration/coco_unlabled_2017_to_firefly.csv",
            "calibration/FlickrFace_to_firefly.csv"],
    "max": ["calibration/all_to_max.csv"]
}

def get_sorted_neg_scores(csv_file_list: List[str]) -> List[float]:
    # read the csv with pandas
    neg_scores = []
    for csv_file in csv_file_list:
        df = pd.read_csv(os.path.join(ROOT, csv_file))
        neg_scores.extend(df["score"].tolist())
    neg_scores = np.array(neg_scores)
    neg_scores.sort()
    return neg_scores


def plot_similarity_map(sim_map:torch.Tensor, vmin:float, vmax:float, title, fname:str):
    fig = go.Figure()

    sim_map_enhanced = ndimage.grey_dilation(sim_map.cpu().numpy(), 5)

    n = sim_map_enhanced.shape[0]  # Number of rows
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1/n))

    fig.add_trace(
        go.Heatmap(
            z=sim_map_enhanced,
            x=freqs,
            y=freqs,
            colorscale="Viridis",
            zmin=vmin, zmax=vmax,
            colorbar=dict(len=1.0),
        )
    )
    

    fig.update_traces(showscale=True)

    # Update layout for better visualization
    fig.update_layout(
        title=title,
        title_font=dict(size=24),  # Increase title font size
        font=dict(size=18), 
    )

    fig.write_image(fname, width=1000, height=1000)


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

    model_to_scores = dict()
    
    # 3. compute scores
    print("Detection by specific models:")
    with torch.no_grad():
        for class_name in CLASS_NAMES:
            model: MaskSim = train_models[class_name]
            imgs = imgs.to(device)
            scores : torch.Tensor = model.compute_probs(imgs)
            score = scores.cpu()[0]
            class_full_name = CLASS_FULLNAME_DICT[class_name]
            print("  Raw score from the detector of {}: {}".format(class_full_name, score))

            model_to_scores[class_name] = score

            neg_scores = get_sorted_neg_scores(NEG_SCORE_FNAME_DICT[class_name])
            n_quantile = np.searchsorted(neg_scores, score)
            fpr = 1 - n_quantile / len(neg_scores)
            print("  FPR corresponding to the score: {:.1f}%".format(fpr * 100))

            if fpr < 0.01:
                print("  It is VERY LIKELY that the image is generated by {}".format(class_full_name))
            elif fpr < 0.05:
                print("  It is LIKELY that the image is generated by {}".format(class_full_name))
            elif fpr < 0.1:
                print("  It is POSSIBLE that the image is generated by {}".format(class_full_name))
            else:
                print("  There is no evidence that the image is generated by {}".format(class_full_name))
            print()

            # visualize the similarity map
            similarity_map : torch.Tensor = model.compute_similarity_map(imgs).detach().cpu().mean(dim=(0,1))
            plot_similarity_map(similarity_map, VISUALIZATION_RANGE[class_name][1], VISUALIZATION_RANGE[class_name][0], class_full_name,  f"sim_map_{class_name}.png")
    print()
    
    print("Detection by generic model")
    max_score = max(model_to_scores.values())
    neg_scores = get_sorted_neg_scores(NEG_SCORE_FNAME_DICT["max"])
    n_quantile = np.searchsorted(neg_scores, max_score)
    fpr = 1 - n_quantile / len(neg_scores)
    print("  Raw score from the generic model: {}".format(max_score))
    print("  FPR corresponding to the score: {:.1f}%".format(fpr * 100))

    if fpr < 0.01:
        print("  It is VERY LIKELY that the image is generated")
    elif fpr < 0.05:
        print("  It is LIKELY that the image is generated")
    elif fpr < 0.1:
        print("  It is POSSIBLE that the image is generated")
    else:
        print("  There is no evidence that the image is generated")
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
