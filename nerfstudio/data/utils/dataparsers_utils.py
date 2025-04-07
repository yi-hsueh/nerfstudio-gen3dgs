# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data parser utils for nerfstudio datasets."""

import math
import os
import json
from typing import List, Tuple

import numpy as np

def get_train_eval_split_sparse(image_filenames: List, N_sparse: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on N views for training and the remaining views for evaluation.
    This follows the approach of the 3DGS-Enhancer(NIPS 2024) paper

    Args:
        image_filenames: list of image filenames
        N_sparse: number of images to use for training
    """
    
    # test if image_filenames is sorted
    assert all(image_filenames[i] <= image_filenames[i + 1] for i in range(len(image_filenames) - 1)), "image_filenames should be sorted"

    '''    
    #Steps:
    Test Views: From the sorted camera list, every 8th view is selected as a test view
    Input Views: The remaining views are sub-sampled using np.linspace to deterministically select exactly N input views
    '''
    num_images = len(image_filenames)
    i_all = np.arange(num_images)
    i_test_for_3DGSEnhancer = i_all[i_all % 8 == 0] # every 8th view is selected as a test view
    i_train_candidates = i_all[i_all % 8 != 0] # exclude every 8th view, which are originally used as eval views, but we will use all remaining views as eval views
    # select N_sparse views from the remaining views
    idx_train = np.linspace(0, len(i_train_candidates) - 1, N_sparse)
    idx_train = [round(i) for i in idx_train]
    i_train = i_train_candidates[idx_train]
    i_eval = np.setdiff1d(i_all, i_train)  # Remaining indices for evaluation
    assert len(i_train) == N_sparse

    print(f"Split dataset into {N_sparse} training and {num_images - N_sparse} evaluation images.")

    # store to JSON file
    parent_dir = os.path.dirname(os.path.dirname(image_filenames[0]))
    split_file = os.path.join(parent_dir, f"split_{str(N_sparse)}_sparse.json")
    split_data = {"train": i_train.tolist(), "eval": i_eval.tolist(), "3DGS_Enhancer_Test": i_test_for_3DGSEnhancer.tolist()}
    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=4)
    print(f"Saved train/eval split to {split_file}")

    return i_train, i_eval


def get_train_eval_split_indices(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the split.json file.
    e.g. split.json file contains:
    {
        "train": [0, 1, 3, 5, 7, 9],
        "eval": [2, 4, 6, 8]
    }
    if "eval": [-1] is used, then the eval set will be all the images not in the train set.

    Args:
        image_filenames: list of image filenames
    """

    # test if image_filenames is sorted
    assert all(image_filenames[i] <= image_filenames[i + 1] for i in range(len(image_filenames) - 1)), "image_filenames should be sorted"
    
    # Load the split.json file
    parent_dir = os.path.dirname(os.path.dirname(image_filenames[0]))
    split_file = os.path.join(parent_dir, "split.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"split.json file not found at {split_file}")
    
    with open(split_file, "r") as f:
        split_data = json.load(f)
    
    num_images = len(image_filenames)
    i_all = np.arange(num_images)
    
    # Get train indices from the JSON file; raise an error if not provided.
    train_indices = split_data.get("train")
    if train_indices is None:
        raise ValueError("split.json must contain a 'train' key")
    
    # Get eval indices; if eval is set to [-1], use the complement of train_indices.
    eval_indices = split_data.get("eval", [])
    if eval_indices == [-1]:
        eval_indices = list(set(i_all) - set(train_indices))
        eval_indices.sort()  # maintain sorted order
    # Otherwise, use the provided eval indices directly.
    
    # Convert the indices lists to numpy arrays
    i_train = np.array(train_indices)
    i_eval = np.array(eval_indices)
    
    # Optional: validate that the indices are within the valid range.
    if not all(0 <= idx < num_images for idx in i_train):
        raise ValueError("Some train indices are out of bounds")
    if not all(0 <= idx < num_images for idx in i_eval):
        raise ValueError("Some eval indices are out of bounds")

    print(f"Loaded train/eval split from {split_file}")
    
    return i_train, i_eval


def get_train_eval_split_fraction_random(image_filenames: List, train_split_fraction: float, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split fraction based on the number of images and the train split fraction, using random sampling.

    Args:
        image_filenames: list of image filenames
        train_split_fraction: fraction of images to use for training
        seed: optional seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)  # Set random seed for reproducibility
    else:
        print('\033[91m' + "Arbitrary random seed for splitting dataset generated!" + '\033[0m')

    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images

    # Generate random indices for training
    i_all = np.arange(num_images)
    i_train = np.random.choice(i_all, size=num_train_images, replace=False)
    i_eval = np.setdiff1d(i_all, i_train)  # Remaining indices for evaluation
    assert len(i_eval) == num_eval_images

    print(f"Randomly split dataset into {num_train_images} training and {num_eval_images} evaluation images.", f" Random seed: {seed}" if seed is not None else "")

    return i_train, i_eval


def get_train_eval_split_fraction(image_filenames: List, train_split_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split fraction based on the number of images and the train split fraction.

    Args:
        image_filenames: list of image filenames
        train_split_fraction: fraction of images to use for training
    """

    # filter image_filenames and poses based on train/eval split percentage
    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images

    return i_train, i_eval


def get_train_eval_split_filename(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the filename of the images.

    Args:
        image_filenames: list of image filenames
    """

    num_images = len(image_filenames)
    basenames = [os.path.basename(image_filename) for image_filename in image_filenames]
    i_all = np.arange(num_images)
    i_train = []
    i_eval = []
    for idx, basename in zip(i_all, basenames):
        # check the frame index
        if "train" in basename:
            i_train.append(idx)
        elif "eval" in basename:
            i_eval.append(idx)
        else:
            raise ValueError("frame should contain train/eval in its name to use this eval-frame-index eval mode")

    return np.array(i_train), np.array(i_eval)


def get_train_eval_split_interval(image_filenames: List, eval_interval: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the interval of the images.

    Args:
        image_filenames: list of image filenames
        eval_interval: interval of images to use for eval
    """

    num_images = len(image_filenames)
    all_indices = np.arange(num_images)
    train_indices = all_indices[all_indices % eval_interval != 0]
    eval_indices = all_indices[all_indices % eval_interval == 0]
    i_train = train_indices
    i_eval = eval_indices

    return i_train, i_eval


def get_train_eval_split_all(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split where all indices are used for both train and eval.

    Args:
        image_filenames: list of image filenames
    """
    num_images = len(image_filenames)
    i_all = np.arange(num_images)
    i_train = i_all
    i_eval = i_all
    return i_train, i_eval
