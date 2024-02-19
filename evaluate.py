import os
import glob
from tqdm import tqdm
from typing import List, Dict
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

import skimage
import pandas as pd

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
        "sd1":"stable-diffusion-1",
        "sd2":"stable-diffusion-2",
        "sdxl":"stable-diffusion-xl",
        "dalle2":"dalle2",
        "dalle3":"dalle3",
        "midjourney": "midjourney",
        "firefly": "firefly",
    }
}

TRAINING_SET = "newsynth"
TEST_SET = "synthbuster"

TRAIN_CLASS_NAMES = [
    "sd1",
    "sd2",
    "sdxl",
    "dalle2",
    "dalle3",
    "midjourney",
    "firefly",
]

TEST_CLASS_NAMES = [
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
BATCH_SZ = 8
NUM_WORKERS = 8  # 6


def print_info(*text):
    print("\033[1;32m" + " ".join(map(str, text)) + "\033[0;0m")


def test_on_all_classes(Q:int):
    """ Train detection model for all the classes of synthetic images.

    Parameters
    ----------
    Q : int
        compression quality factor
    """

    compress_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    color_space = "RGB" if Q is None else "YCbCr"

    score_folder = f"output/score/{compress_tag}/{TRAINING_SET}"
    os.makedirs(score_folder, exist_ok=True)

    device = torch.device("cuda:0")

    # real_test_folder = f"processed_data/{compress_tag}/dresden"

    # new try
    real_test_folder = f"processed_data/{compress_tag}/raise-2k"
    
    # load all trained models
    train_models: Dict[str, MaskSim] = dict()
    for train_class_name in TRAIN_CLASS_NAMES:
        ckpt_fpattern = os.path.join(f"checkpoints/{compress_tag}/{TRAINING_SET}", train_class_name + "*.ckpt")
        ckpt_fnames = glob.glob(ckpt_fpattern)
        ckpt_fnames.sort(key=os.path.getctime)
        assert len(ckpt_fnames) > 0, "Cannot find checkpoints in " + os.path.join(f"checkpoints/{compress_tag}")

        ckpt_fname = ckpt_fnames[-1]  # take the latest version that corresponds to the pattern
        model = MaskSim.load_from_checkpoint(ckpt_fname, map_location=device).float()
        print_info("Loaded model from ", ckpt_fname)

        model.eval()
        model.freeze()
        train_models[train_class_name] = model

        masks = model.get_mask()
        for i in range(len(masks)):
            mask = masks[i].detach().cpu()
            mask_save_dir = f"output/mask/{compress_tag}/{TRAINING_SET}"
            os.makedirs(mask_save_dir, exist_ok=True)
            skimage.io.imsave(os.path.join(mask_save_dir, f"{train_class_name}_{i:02d}.tiff"),
                            torch.permute(mask, (1, 2, 0)).numpy()[:, :, :3])
            print_info(f"Mask is saved in:", os.path.join(mask_save_dir, f"{train_class_name}_{i:02d}.tiff"))

            spec_ref = model.ref_pattern_list[i].detach().cpu()
            ref_save_dir = f"output/ref/{compress_tag}/{TRAINING_SET}"
            os.makedirs(ref_save_dir, exist_ok=True)
            skimage.io.imsave(os.path.join(ref_save_dir, f"{train_class_name}_{i:02d}.tiff"),
                            torch.permute(spec_ref, (1, 2, 0)).numpy()[:, :, :3])
            print_info(f"Spectrum reference is saved in:", os.path.join(ref_save_dir, f"{train_class_name}_{i:02d}.tiff"))


    test_dataset = MaskSimDataset(img_dir_real_list=[real_test_folder], 
                                  img_dir_fake_list=[],
                                  img_size=IMG_SIZE,
                                  channels=CHANNELS,
                                  color_space=color_space,
                                  mode="test",
                                  limit_nb_img=1488,
                                  )
    real_test_dataloader = utils.data.DataLoader(test_dataset, batch_size=BATCH_SZ,
                                                 shuffle=False, num_workers=NUM_WORKERS,
                                                 collate_fn=MaskSimDataset.collate_fn)

    real_to_fake_model_scores = dict()
    for train_class_name in TRAIN_CLASS_NAMES:
        real_to_fake_model_scores[train_class_name] = []

    for batch in tqdm(real_test_dataloader):
        imgs, labels = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            for train_class_name in TRAIN_CLASS_NAMES:
                model = train_models[train_class_name]
                scores : torch.Tensor = model.compute_probs(imgs)
                real_to_fake_model_scores[train_class_name].append(scores.detach().cpu().numpy())

    for train_class_name in TRAIN_CLASS_NAMES:
        real_to_fake_model_scores[train_class_name] = np.concatenate(real_to_fake_model_scores[train_class_name])
        print_info(real_to_fake_model_scores[train_class_name].shape)
        save_fname = os.path.join(score_folder, f"real_to_{train_class_name}.txt")
        np.savetxt(save_fname, real_to_fake_model_scores[train_class_name])
        print_info(f"Saved real to fake proba in {save_fname}")

    for class_idx, test_class_name in enumerate(TEST_CLASS_NAMES):
        fake_test_folder = f"processed_data/{compress_tag}/{TEST_SET}/{DATASETS[TEST_SET][test_class_name]}"
        test_dataset = MaskSimDataset(img_dir_real_list=[], 
                                        img_dir_fake_list=[fake_test_folder],
                                        img_size=IMG_SIZE,
                                        channels=CHANNELS,
                                        color_space=color_space,
                                        mode="test",
                                        )
        test_dataloader = utils.data.DataLoader(test_dataset, batch_size=BATCH_SZ,
                                                shuffle=False, num_workers=NUM_WORKERS,
                                                collate_fn=MaskSimDataset.collate_fn)

        fake_to_fake_model_scores = dict()
        for train_class_name in TRAIN_CLASS_NAMES:
            fake_to_fake_model_scores[train_class_name] = []

        for batch in tqdm(test_dataloader):
            imgs, labels = batch
            imgs = imgs.to(device)
            with torch.no_grad():
                for train_class_name in TRAIN_CLASS_NAMES:
                    model = train_models[train_class_name]
                    scores : torch.Tensor = model.compute_probs(imgs)
                    fake_to_fake_model_scores[train_class_name].append(scores.detach().cpu().numpy())

        for train_class_name in TRAIN_CLASS_NAMES:
            fake_to_fake_model_scores[train_class_name] = np.concatenate(fake_to_fake_model_scores[train_class_name])
            print_info(fake_to_fake_model_scores[train_class_name].shape)
            save_fname = os.path.join(score_folder, f"{test_class_name}_to_{train_class_name}.txt")
            np.savetxt(save_fname, fake_to_fake_model_scores[train_class_name])
            print_info(f"Saved fake to fake proba in {save_fname}")
    
    print_info(f"Finished `compute_scores_one_class` with {train_class_name}")


def merge_scores(test_class_name:str, Q:int, excluded_class_for_merge:str=None):
    """ Merge scores of different models by maximum

    Parameters
    ----------
    test_class_name : str
        the class of tested images
    Q : int
        JPEG compression quality factor
    excluded_class_for_merge : bool
        The merged detector excludes the model of the class
    """

    except_suffix = ""
    if excluded_class_for_merge is not None:
        except_suffix = f"_except_{excluded_class_for_merge}"

    compress_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"

    similarity_folder = f"output/score/{compress_tag}/{TRAINING_SET}"

    statistics_folder = f"output/statistics/{compress_tag}/{TRAINING_SET}"
    os.makedirs(statistics_folder, exist_ok=True)
    
    # for each test class, load the scores w.r.t. 7 (or 6 for leave_one_out) models
    test_to_all_models_scores = []
    for train_class_name in TRAIN_CLASS_NAMES:
        if excluded_class_for_merge is not None and train_class_name == excluded_class_for_merge:
            continue

        test_to_one_model_scores_fname = os.path.join(similarity_folder, f"{test_class_name}_to_{train_class_name}.txt")
        test_to_one_model_scores = np.loadtxt(test_to_one_model_scores_fname)

        test_to_all_models_scores.append(test_to_one_model_scores)
    test_to_all_models_scores = np.stack(test_to_all_models_scores)
    print_info(test_to_all_models_scores.shape) # 7, 1000

    probas = np.max(test_to_all_models_scores, axis=0) # 1000

    save_fname = os.path.join(statistics_folder, f"{test_class_name}_to_all_probas{except_suffix}.txt")
    np.savetxt(save_fname, probas)
    print_info(f"Saved probas for {test_class_name} in {save_fname}. ")


def _compute_statistics_merged(leave_one_out:bool):
    
    from metrics_util import auroc, acc, mcc, mcc_best
    stats_metrics = {
        "auroc": auroc,
        "acc": acc,
        "mcc": mcc,
        "mcc_best": mcc_best,
    }
    
    compress_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    statistics_folder = f"output/statistics/{compress_tag}/{TRAINING_SET}"

    results_data = {
        "test_class": [ DATASETS[TEST_SET][class_name] for class_name in  TEST_CLASS_NAMES ],
    }
    for metric_name in stats_metrics.keys():
        results_data[metric_name] = list()

    for test_class_name in TEST_CLASS_NAMES:
        except_suffix = ""
        if leave_one_out:
            except_suffix = f"_except_{test_class_name}"

        scores_real_fname = os.path.join(statistics_folder, f"real_to_all_probas" +  except_suffix + ".txt")
        scores_real = np.loadtxt(scores_real_fname)

        scores_fake_fname = os.path.join(statistics_folder, f"{test_class_name}_to_all_probas" + except_suffix + ".txt")
        scores_fake = np.loadtxt(scores_fake_fname)

        for metric_name, metric_func in stats_metrics.items():
            res = metric_func(scores_real, scores_fake) * 100
            results_data[metric_name].append(res)

    df = pd.DataFrame(results_data)
    avg_row = df.mean(numeric_only=True)
    avg_row["test_class"] = "AVG"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    print_info(df)



def compute_statistics_generalized():
    return _compute_statistics_merged(leave_one_out=True)


def compute_statistics_generic():
    return _compute_statistics_merged(leave_one_out=False)

def compute_statistics_sd1():

    from metrics_util import auroc, aucpr, ap, acc, acc_best, mcc, mcc_best
    stats_metrics = {
        "auroc": auroc,
        "acc": acc,
        "mcc": mcc,
        "mcc_best": mcc_best,
    }
    
    compress_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    statistics_folder = f"output/score/{compress_tag}/{TRAINING_SET}"
    
    results_data = {
        "test_class": [ DATASETS[TEST_SET][class_name] for class_name in  TEST_CLASS_NAMES ],
    }
    for metric_name in stats_metrics.keys():
        results_data[metric_name] = list()

    train_class_name = "sd1"
    for test_class_name in TEST_CLASS_NAMES:

        scores_real_fname = os.path.join(statistics_folder, f"real_to_{train_class_name}.txt")
        scores_real = np.loadtxt(scores_real_fname)

        scores_fake_fname = os.path.join(statistics_folder, f"{test_class_name}_to_{train_class_name}.txt")
        scores_fake = np.loadtxt(scores_fake_fname)

        for metric_name, metric_func in stats_metrics.items():
            res = metric_func(scores_real, scores_fake) * 100
            results_data[metric_name].append(res)
    df = pd.DataFrame(results_data)
    avg_row = df.mean(numeric_only=True)
    avg_row["test_class"] = "AVG"
    # df = df.append(avg_row, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    print_info(df)

# def compute_statistics_one_to_one():
    
#     from metrics_util import auroc, aucpr
#     stats_metrics = {
#         "auroc": auroc,
#         "aucpr": aucpr
#     }
    
#     comp_or_uncomp = f"uncompress"

#     # statistics_folder = f"out_statistics/{comp_or_uncomp}/{TRAINING_SET}"

#     proba_folder = f"score/{comp_or_uncomp}/{TRAINING_SET}"

#     results_data = {
#         "test_class": [ DATASETS[TEST_SET][class_name] for class_name in  TEST_CLASS_NAMES ],
#     }
#     for metric_name in stats_metrics.keys():
#         results_data[metric_name] = np.zeros((len(TEST_CLASS_NAMES), len(TRAIN_CLASS_NAMES)))

    
#     for col, train_class_name in enumerate(TRAIN_CLASS_NAMES):
#         scores_real_fname = os.path.join(proba_folder, f"real_to_{train_class_name}.txt")
#         scores_real = np.loadtxt(scores_real_fname)

#         for row, test_class_name in enumerate(TEST_CLASS_NAMES):
#             scores_fake_fname = os.path.join(proba_folder, f"{test_class_name}_to_{train_class_name}.txt")
#             scores_fake = np.loadtxt(scores_fake_fname)

#             for metric_name, metric_func in stats_metrics.items():
#                 res = metric_func(scores_real, scores_fake) * 100
#                 results_data[metric_name][row, col] = res
    

#     df_auroc = pd.DataFrame(results_data["auroc"])

#     avg_row = df_auroc.mean().to_frame().T
    
#     df_auroc = pd.concat([df_auroc, avg_row], ignore_index=True, axis=0)
#     print_info(df_auroc)

#     df_auroc.columns = [DATASETS[TRAINING_SET][class_name] for class_name in TRAIN_CLASS_NAMES]
#     df_auroc.index = [DATASETS[TEST_SET][class_name] for class_name in TEST_CLASS_NAMES] + ["AVG"]
#     print_info("AUROC:")
#     print_info(df_auroc.to_latex(float_format='%.1f'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MaskSim: Detection of synthetic images by masked spectrum similarity analysis. (c) 2024 Yanhao Li. Under license GNU AGPL.'
    )

    parser.add_argument('-Q', '--Q', type=int, required=False,
                        help='compression quality factor', default=None)
    args = parser.parse_args()

    Q = args.Q

    # step 1: predict scores
    test_on_all_classes(Q=Q)

    # step 2: merge scores
    merge_scores(test_class_name="real", Q=Q)
    for test_class_name in TEST_CLASS_NAMES:
        merge_scores(test_class_name, Q=Q)
        merge_scores(test_class_name="real", Q=Q, excluded_class_for_merge=test_class_name)
        merge_scores(test_class_name=test_class_name, Q=Q, excluded_class_for_merge=test_class_name)

    # step 3: compute different statistics
    # step 3.1 trained on SD-1
    print_info("Detector trained on Stable Diff. 1:")
    compute_statistics_sd1()
    print()

    # step 3.2 generalized
    print_info("Generalized detector:")
    compute_statistics_generalized()
    print()

    # step 3.3 generic
    print_info("Generic detector:")
    compute_statistics_generic()
    print()
