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

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from torch.utils.data import IterableDataset

from cosmos_datasets.webdataset_base.augmentors.augmentor import Augmentor


@dataclass
class DatasetInfo:
    wdinfo: List[str]  # List of wdinfo files
    opts: Dict[str, Any] = field(default_factory=dict)  # Additional dataset info args
    per_dataset_keys: List[str] = field(default_factory=list)  # List of keys per dataset
    source: str = ""  # data source


@dataclass
class TarSample:
    path: str  # Path to the sample
    root: str  # Root folder
    keys: List[str]  # List of keys to be loaded from the webdataset
    meta: DatasetInfo  # Metadata
    dset_id: str  # Dataset id
    sample_keys_full_list: Optional[str] = None  # Path to the file containing full sample keys for the tar file


@dataclass
class Wdinfo:
    tar_files: List[TarSample]  # List of all tar samples
    total_key_count: int  # Total number of elements present in the dataset
    chunk_size: int  # Number of elements present in each tar


@dataclass
class AugmentorConfig:
    # Type of augmentor
    type: type[Augmentor]
    # Input keys used by the augmentor
    input_keys: List[str]
    # Output keys returned by the augmentor
    output_keys: Optional[List[str]] = None
    # Additional arguments used by the augmentor
    args: Optional[Dict[str, Any]] = None

    def make_instance(self) -> Augmentor:
        return self.type(input_keys=self.input_keys, output_keys=self.output_keys, args=self.args)


@dataclass
class DatasetConfig:
    keys: List[str]  # List of keys used
    buffer_size: int  # Buffer size used by each worker
    dataset_info: List[DatasetInfo]  # List of dataset info files, one for each dataset
    distributor: IterableDataset  # Iterator for returning list of tar files
    decoders: List[Any]  # List of decoder functions for decoding bytestream
    augmentation: Dict[str, AugmentorConfig]  # Dictionary containing all augmentations
    streaming_download: bool = True  # Whether to use streaming loader
    remove_extension_from_keys: bool = True  # True: objects will have a key of data_type; False: data_type.extension
    sample_keys_full_list_path: Optional[str] = None  # Path to the file containing all keys present in the dataset, e.g., "index"
