# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos Datasets - A standalone dataset library for machine learning workflows."""

__version__ = "1.0.0"

# Import main dataset providers
from .dataset_provider import get_video_dataset

from . import s3_support

# Import key classes
from .webdataset import Dataset
from .joint_dataloader import IterativeJointDataLoader, RandomJointDataLoader
from .cached_replay_dataloader import CachedReplayDataLoader

# Import utilities
from .constants import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO
from .augmentor_provider import AUGMENTOR_OPTIONS

from .s3_support import configure_s3_defaults

__all__ = [
    "get_video_dataset",
    "Dataset",
    "IterativeJointDataLoader",
    "RandomJointDataLoader", 
    "CachedReplayDataLoader",
    "IMAGE_RES_SIZE_INFO",
    "VIDEO_RES_SIZE_INFO", 
    "AUGMENTOR_OPTIONS",
    "configure_s3_defaults",
]
