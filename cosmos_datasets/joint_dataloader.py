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


import numpy as np
import torch
import webdataset

from cosmos_datasets.lazy_config import instantiate


class IterativeJointDataLoader(webdataset.WebLoader):
    r"""
    A joint dataloader that supports loading both images and videos.
    """

    def __init__(self, dataloaders: dict[str, dict[str, any]]):
        """
        Initialize the JointDataLoader with multiple datasets.

        Args:
            dataloaders: key - dataset_name; value - {"dataloader": config_object, "ratio": data_ratio}
                        where config_object is a lazy configuration that will be instantiated

        Example:
            joint_loader = IterativeJointDataLoader(
                dataloaders{
                    "image_data": {
                        "dataloader": L(get_cached_replay_dataloader)(...),
                        "ratio": 4,
                    },
                    "video_data": {
                        "dataloader": L(get_video_dataloader)(...),
                        "ratio": 1,
                    },
                }
            )
        """
        self.dataloader_list, self.dataset_name_list, self.data_ratios = [], [], []

        for dataset_name, dataloader_data in dataloaders.items():
            assert set(dataloader_data.keys()) == {"dataloader", "ratio"}, f"Invalid config: {dataloader_data}"
            self.dataset_name_list.append(dataset_name)
            
            # Instantiate the dataloader from config
            dataloader = dataloader_data["dataloader"]
            self.dataloader_list.append(instantiate(dataloader))
            
            self.data_ratios.append(dataloader_data["ratio"])

        self.global_id = 0
        self.ratio_sum = sum(self.data_ratios)

        self.data_len = 0
        self.dataloaders = [iter(dataloader) for dataloader in self.dataloader_list]
        for data in self.dataloader_list:
            print(f"DEBUG: Dataloader type: {type(data)}")
            print(f"DEBUG: Has __len__: {hasattr(data, '__len__')}")
            if hasattr(data, '__len__'):
                try:
                    length = len(data)
                    print(f"DEBUG: Length: {length}")
                    self.data_len += length
                except Exception as e:
                    print(f"DEBUG: Error getting length: {e}")
                    self.data_len += 1  # Fallback
            else:
                self.data_len += 1  # Fallback for dataloaders without length

    def __len__(self) -> int:
        return self.data_len

    def __iter__(self):
        while True:
            data_id = self.global_id % self.ratio_sum
            index_id = self._get_dataloader_index(data_id)
            curr_dataloader = self.dataloaders[index_id]
            batch = next(curr_dataloader)
            
            # Handle batched output - if it's a list, get the first item
            if isinstance(batch, list) and len(batch) > 0:
                output = batch[0]
            else:
                output = batch
                
            output["dataset_name"] = self.dataset_name_list[index_id]
            self.global_id += 1
            del curr_dataloader
            yield output

    def _get_dataloader_index(self, data_id):
        """Maps global id to the corresponding dataloader index based on ratio."""
        for i, r in enumerate(self.data_ratios):
            if data_id < r:
                return i
            data_id -= r
        raise ValueError("Invalid data_id")


class RandomJointDataLoader(webdataset.WebLoader):
    r"""
    A joint dataloader that supports randomly samples batches from multiple datasets.
    """

    # def __init__(self, **kwargs):
    def __init__(self, dataloaders: dict[str, dict[str, any]]):
        """
        Initialize the RandomJointDataLoader with multiple datasets.

        Args:
            dataloaders: key - dataset_name; value - {"dataloader": config_object, "ratio": sample_probability}
                        where config_object is a lazy configuration that will be instantiated
                        and sample_probability is the probability of sampling from this dataset

        Example:
            joint_loader = RandomJointDataLoader(
                dataloaders={
                    "image_data": {
                        "dataloader": L(get_cached_replay_dataloader)(...),
                        "ratio": 0.8,
                    },
                    "video_data": {
                        "dataloader": L(get_video_dataloader)(...),
                        "ratio": 0.2,
                    },
                }
            )
        """
        self.dataloader_list, self.dataset_name_list, self.data_ratios = [], [], []

        for dataset_name, dataloader_data in dataloaders.items():
            assert set(dataloader_data.keys()) == {"dataloader", "ratio"}, f"Invalid config: {dataloader_data}"
            self.dataset_name_list.append(dataset_name)
            
            # Instantiate the dataloader from config
            dataloader = dataloader_data["dataloader"]
            self.dataloader_list.append(instantiate(dataloader))
            
            self.data_ratios.append(dataloader_data["ratio"])

        assert sum(self.data_ratios) == 1.0, "Sum of sample probabilities should be equal to 1."

        self.data_len = 0
        self.dataloaders = [iter(dataloader) for dataloader in self.dataloader_list]
        for data in self.dataloader_list:
            self.data_len += len(data)

    def __len__(self) -> int:
        return self.data_len

    def __iter__(self):
        while True:
            # Sample a random dataset
            data_id = int(np.random.choice(len(self.dataloader_list), 1, p=self.data_ratios)[0])
            curr_dataloader = self.dataloaders[data_id]
            output = next(curr_dataloader)
            output["dataset_name"] = self.dataset_name_list[data_id]
            del curr_dataloader
            yield output
