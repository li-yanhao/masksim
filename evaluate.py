import json
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
REAL_TEST_SET = "raise"

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
NUM_WORKERS = 8


def print_info(*text):
    print("\033[1;32m" + " ".join(map(str, text)) + "\033[0;0m")


def test_on_all_classes(Q:str):
    """ Train detection model for all the classes of synthetic images.

    Parameters
    ----------
    Q : int
        compression quality factor
    """

    if Q is None:
        ckpt_tag = "uncompressed"
    else:
        ckpt_tag = f"JPEG_Q{Q}"
    ckpt_tag += f"_w{IMG_SIZE}"


    if Q_TEST is None:
        data_tag = "uncompressed"
    else:
        if COMPRESSION_TYPE == "jpeg":
            data_tag = f"JPEG_Q{Q_TEST}"   
        elif COMPRESSION_TYPE == "webp":
            data_tag = f"WEBP_Q{Q_TEST}"   

    color_space = "RGB" if Q is None else "YCbCr"

    if VERSION != "":
        version_tag = "_" + VERSION
    else:
        version_tag = ""

    score_folder = f"output/score/{ckpt_tag}/{data_tag}/{TRAINING_SET}"
    os.makedirs(score_folder, exist_ok=True)

    device = torch.device("cuda:0")

    real_test_folder = f"processed_data/{data_tag}/{REAL_TEST_SET}"
    
    # load all trained models
    train_models: Dict[str, MaskSim] = dict()
    for train_class_name in TRAIN_CLASS_NAMES:
        ckpt_fpattern = os.path.join(f"checkpoints{version_tag}/{ckpt_tag}/{TRAINING_SET}", train_class_name + "*.ckpt")
        ckpt_fnames = glob.glob(ckpt_fpattern)
        ckpt_fnames.sort(key=os.path.getctime)
        assert len(ckpt_fnames) > 0, "Cannot find checkpoints in " + os.path.join(f"checkpoints{version_tag}/{ckpt_tag}")

        ckpt_fname = ckpt_fnames[-1]  # take the latest version that corresponds to the pattern
        model = MaskSim.load_from_checkpoint(ckpt_fname, map_location=device).float()
        print_info("Loaded model from ", ckpt_fname)

        model.eval()
        model.freeze()
        train_models[train_class_name] = model

        masks = model.get_mask()
        for i in range(len(masks)):
            mask = masks[i].detach().cpu()
            mask_save_dir = f"output/mask/{ckpt_tag}/{TRAINING_SET}"
            os.makedirs(mask_save_dir, exist_ok=True)
            skimage.io.imsave(os.path.join(mask_save_dir, f"{train_class_name}_{i:02d}.tiff"),
                            torch.permute(mask, (1, 2, 0)).numpy()[:, :, :3])
            print_info(f"Mask is saved in:", os.path.join(mask_save_dir, f"{train_class_name}_{i:02d}.tiff"))

            spec_ref = model.ref_pattern_list[i].detach().cpu()
            ref_save_dir = f"output/ref/{ckpt_tag}/{TRAINING_SET}"
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
                                  return_fname=True
                                  )
    real_test_dataloader = utils.data.DataLoader(test_dataset, batch_size=BATCH_SZ,
                                                 shuffle=False, num_workers=NUM_WORKERS,
                                                 collate_fn=MaskSimDataset.collate_fn)

    real_to_fake_model_scores = dict()
    for train_class_name in TRAIN_CLASS_NAMES:
        real_to_fake_model_scores[train_class_name] = list()

    for batch in tqdm(real_test_dataloader):
        imgs, labels, fnames = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            for train_class_name in TRAIN_CLASS_NAMES:
                model = train_models[train_class_name]
                scores : torch.Tensor = model.compute_probs(imgs)
                scores = scores.detach().cpu().numpy()

                for fname, score in zip(fnames, scores):
                    real_to_fake_model_scores[train_class_name].append((fname, score.item()))

    # save txt
    for train_class_name in TRAIN_CLASS_NAMES:
        scores = [item[1] for item in real_to_fake_model_scores[train_class_name]]
        scores = np.array(scores)
        save_fname = os.path.join(score_folder, f"real_to_{train_class_name}.txt")
        np.savetxt(save_fname, scores)
        print_info(f"Saved real to fake proba in {save_fname}")

    # save json
    for train_class_name in TRAIN_CLASS_NAMES:
        save_fname = os.path.join(score_folder, f"real_to_{train_class_name}.json")
        with open(save_fname, 'w') as f:
            json.dump(real_to_fake_model_scores[train_class_name], f, indent=2)
        print_info(f"Saved real to fake proba in {save_fname}")


    for class_idx, test_class_name in enumerate(TEST_CLASS_NAMES):
        fake_test_folder = f"processed_data/{data_tag}/{TEST_SET}/{DATASETS[TEST_SET][test_class_name]}"
        test_dataset = MaskSimDataset(img_dir_real_list=[], 
                                        img_dir_fake_list=[fake_test_folder],
                                        img_size=IMG_SIZE,
                                        channels=CHANNELS,
                                        color_space=color_space,
                                        mode="test",
                                        return_fname=True,
                                        )
        test_dataloader = utils.data.DataLoader(test_dataset, batch_size=BATCH_SZ,
                                                shuffle=False, num_workers=NUM_WORKERS,
                                                collate_fn=MaskSimDataset.collate_fn)

        fake_to_fake_model_scores = dict()
        for train_class_name in TRAIN_CLASS_NAMES:
            fake_to_fake_model_scores[train_class_name] = list()

        for batch in tqdm(test_dataloader):
            imgs, labels, fnames = batch
            imgs = imgs.to(device)
            with torch.no_grad():
                for train_class_name in TRAIN_CLASS_NAMES:
                    model = train_models[train_class_name]
                    scores : torch.Tensor = model.compute_probs(imgs)
                    scores = scores.detach().cpu().numpy()

                    for fname, score in zip(fnames, scores):
                        fake_to_fake_model_scores[train_class_name].append((fname, score.item()))
        
        for train_class_name in TRAIN_CLASS_NAMES:
            scores = [item[1] for item in fake_to_fake_model_scores[train_class_name]]
            scores = np.array(scores)
            save_fname = os.path.join(score_folder, f"{test_class_name}_to_{train_class_name}.txt")
            np.savetxt(save_fname, scores)
            print_info(f"Saved fake to fake proba in {save_fname}")
        
        # save json
        for train_class_name in TRAIN_CLASS_NAMES:
            save_fname = os.path.join(score_folder, f"{test_class_name}_to_{train_class_name}.json")
            with open(save_fname, 'w') as f:
                json.dump(fake_to_fake_model_scores[train_class_name], f, indent=2)
            print_info(f"Saved fake to fake proba in {save_fname}")

    print_info(f"Finished `test_on_all_classes` with {TEST_CLASS_NAMES}")


def merge_scores(test_class_name:str, Q:str, excluded_class_for_merge:str=None):
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

    ckpt_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    ckpt_tag += f"_w{IMG_SIZE}"

    if Q_TEST is None:
        data_tag = "uncompressed"
    else:
        if COMPRESSION_TYPE == "jpeg":
            data_tag = f"JPEG_Q{Q_TEST}"   
        elif COMPRESSION_TYPE == "webp":
            data_tag = f"WEBP_Q{Q_TEST}"    

    score_folder = f"output/score/{ckpt_tag}/{data_tag}/{TRAINING_SET}"

    statistics_folder = f"output/statistics/{ckpt_tag}/{data_tag}/{TRAINING_SET}"
    os.makedirs(statistics_folder, exist_ok=True)
    
    # for each test class, load the scores w.r.t. 7 (or 6 for leave_one_out) models
    test_to_all_models_scores = []
    for train_class_name in TRAIN_CLASS_NAMES:
        if excluded_class_for_merge is not None and train_class_name == excluded_class_for_merge:
            continue

        test_to_one_model_scores_fname = os.path.join(score_folder, f"{test_class_name}_to_{train_class_name}.txt")
        test_to_one_model_scores = np.loadtxt(test_to_one_model_scores_fname)

        test_to_all_models_scores.append(test_to_one_model_scores)
    test_to_all_models_scores = np.stack(test_to_all_models_scores)

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
    
    ckpt_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    ckpt_tag += f"_w{IMG_SIZE}"

    if Q_TEST is None:
        data_tag = "uncompressed"
    else:
        if COMPRESSION_TYPE == "jpeg":
            data_tag = f"JPEG_Q{Q_TEST}"   
        elif COMPRESSION_TYPE == "webp":
            data_tag = f"WEBP_Q{Q_TEST}"   

    statistics_folder = f"output/statistics/{ckpt_tag}/{data_tag}/{TRAINING_SET}"

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


def compute_statistics_one_class(train_class_name):

    from metrics_util import auroc, aucpr, ap, acc, acc_best, mcc, mcc_best
    stats_metrics = {
        "auroc": auroc,
        "acc": acc,
        "mcc": mcc,
        "mcc_best": mcc_best,
    }
    
    ckpt_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    ckpt_tag += f"_w{IMG_SIZE}"

    if Q_TEST is None:
        data_tag = "uncompressed"
    else:
        if COMPRESSION_TYPE == "jpeg":
            data_tag = f"JPEG_Q{Q_TEST}"   
        elif COMPRESSION_TYPE == "webp":
            data_tag = f"WEBP_Q{Q_TEST}"  

    statistics_folder = f"output/score/{ckpt_tag}/{data_tag}/{TRAINING_SET}"
    
    
    results_data = {
        "test_class": [ DATASETS[TEST_SET][class_name] for class_name in  TEST_CLASS_NAMES ],
    }
    for metric_name in stats_metrics.keys():
        results_data[metric_name] = list()

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
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    print_info(df)


def compute_statistics_specific():
    
    from metrics_util import auroc, acc, mcc, mcc_best
    stats_metrics = {
        "auroc": auroc,
        "acc": acc,
        "mcc": mcc,
        "mcc_best": mcc_best,
    }
    
    ckpt_tag = "uncompressed" if Q is None else f"JPEG_Q{Q}"
    ckpt_tag += f"_w{IMG_SIZE}"

    if Q_TEST is None:
        data_tag = "uncompressed"
    else:
        if COMPRESSION_TYPE == "jpeg":
            data_tag = f"JPEG_Q{Q_TEST}"   
        elif COMPRESSION_TYPE == "webp":
            data_tag = f"WEBP_Q{Q_TEST}"   

    score_folder = f"output/score/{ckpt_tag}/{data_tag}/{TRAINING_SET}"

    results_data = {
        "test_class": [ DATASETS[TEST_SET][class_name] for class_name in  TEST_CLASS_NAMES ],
    }
    for metric_name in stats_metrics.keys():
        results_data[metric_name] = list()

    for test_class_name in TEST_CLASS_NAMES:

        scores_real_fname = os.path.join(score_folder, f"real_to_{test_class_name}.txt")
        scores_real = np.loadtxt(scores_real_fname)

        scores_fake_fname = os.path.join(score_folder, f"{test_class_name}_to_{test_class_name}.txt")
        scores_fake = np.loadtxt(scores_fake_fname)

        for metric_name, metric_func in stats_metrics.items():
            res = metric_func(scores_real, scores_fake) * 100
            results_data[metric_name].append(res)

    df = pd.DataFrame(results_data)
    avg_row = df.mean(numeric_only=True)
    avg_row["test_class"] = "AVG"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    print_info(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MaskSim: Detection of synthetic images by masked spectrum similarity analysis. (c) 2024 Yanhao Li. Under license GNU AGPL.'
    )

    parser.add_argument('-Q', '--Q', type=str, required=False,
                        help='compression quality factor for training', default="90")
    parser.add_argument('--img_q', type=str, required=False,
                        help='compression quality factor for tested images', default=None)
    parser.add_argument('-w', '--w', type=int, required=False,
                        help='image size', default=64)
    parser.add_argument('-v', '--version', type=str, required=False,
                        help='version of checkpoints', default="")
    parser.add_argument('--compression', type=str, required=False,
                        help='compression type: jpeg or webp', default=None)
    args = parser.parse_args()

    Q = args.Q
    IMG_SIZE = args.w
    VERSION = args.version
    Q_TEST = args.img_q
    COMPRESSION_TYPE = args.compression

    # # step 1: predict scores
    test_on_all_classes(Q=Q)

    # step 2: merge scores
    merge_scores(test_class_name="real", Q=Q)
    for test_class_name in TEST_CLASS_NAMES:
        merge_scores(test_class_name, Q=Q)
        merge_scores(test_class_name="real", Q=Q, excluded_class_for_merge=test_class_name)
        merge_scores(test_class_name=test_class_name, Q=Q, excluded_class_for_merge=test_class_name)

    # step 3: compute different statistics
    # step 3.1 trained on SD-2 / other classes
    for train_class_name in TRAIN_CLASS_NAMES:
        print_info(f"Detector trained on {train_class_name}:")
        compute_statistics_one_class(train_class_name=train_class_name)
        print()

    # # step 3.2 generalized
    print_info("Generalized detector:")
    compute_statistics_generalized()
    print()

    # step 3.3 generic
    print_info("Generic detector:")
    compute_statistics_generic()
    print()

    # step 3.4 specific
    print_info("Specific detector:")
    compute_statistics_specific()
    print()
