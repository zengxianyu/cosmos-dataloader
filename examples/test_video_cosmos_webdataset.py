#!/usr/bin/env python3

"""
Test cosmos_datasets webdataset with actual video files.
This creates proper blank video files that can be decoded by the video decoders.
"""

import os
import cosmos_datasets
import sys
import tempfile
import json
import tarfile
import pickle
import subprocess
from io import BytesIO
from pathlib import Path
import logging

# Add the parent directory to sys.path to import cosmos_datasets
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmos_datasets.webdataset_base.dataloader import DataLoader
from cosmos_datasets.log import get_logger
from cosmos_datasets.webdataset import Dataset
from cosmos_datasets.webdataset_base.config.schema import DatasetConfig, DatasetInfo
from cosmos_datasets.webdataset_base.distributors.basic import ShardlistBasic
from cosmos_datasets.webdataset_base.distributors.multi_aspect_ratio import ShardlistMultiAspectRatio
from cosmos_datasets.webdataset_base.decoders.video_decoder import construct_video_decoder
from examples.dataset_utils import create_multi_aspect_ratio_video_dataset, create_video_dataset_with_real_videos
from webdataset.handlers import warn_and_continue
from cosmos_datasets.augmentor_provider import get_video_augmentor_v2
from cosmos_datasets.webdataset_base.decoders.video_decoder import video_naive_bytes

logger = get_logger(__name__)

def get_example_video_dataset(is_infinite: bool = False, s3_bucket: str = None, s3_profile: str = None, s3_endpoint_url: str = None):
    # Create test data with videos in multi-modal structure
    wdinfo_path, dataset_info = create_video_dataset_with_real_videos("examples/video_test_data", s3_bucket=s3_bucket, s3_profile=s3_profile, s3_endpoint_url=s3_endpoint_url)

    # Create distributor
    distributor = ShardlistBasic(
        shuffle=False,
        split_by_node=False,
        split_by_worker=False,
        resume_flag=True,
        verbose=False,
        is_infinite_loader=is_infinite,
        max_epochs=1,
        repeat_url=False
    )
    
    # Create config with video decoder and proper error handler
    config = DatasetConfig(
        keys=[],
        buffer_size=100,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[video_naive_bytes()],
        augmentation=get_video_augmentor_v2(resolution="512", num_video_frames=60), 
        # shorter video (i.e. the caption window) will be padded to num_video_frames
        # videos shorter than min_duration or fps does not fall in the range will be skipped
        remove_extension_from_keys=True,
        sample_keys_full_list_path=None,
    )
        
    dataset = Dataset(config=config, decoder_handler=warn_and_continue, detshuffle=True)
    return dataset


def test_video_cosmos_dataset(s3_bucket: str = None, s3_profile: str = None, s3_endpoint_url: str = None):
    """Test cosmos_datasets with actual video data"""
    
    logger.info("=== Video cosmos_datasets Test ===")

    built_dataset = get_example_video_dataset(s3_bucket=s3_bucket, s3_profile=s3_profile, s3_endpoint_url=s3_endpoint_url).build_dataset()
    logger.info(f"✓ Built dataset with video_naive_bytes decoder")
    logger.info(f"  Total samples: {getattr(built_dataset, 'total_images', 'N/A')}")
        
    # Process samples
    sample_count = 0
    for sample in built_dataset:
        logger.info(f"  Keys: {list(sample.keys())}")
        logger.info(f" video size: {sample['video'].shape}")
        logger.info(f"Sample metas: {sample['metas'] }")
        
        sample_count += 1
    
    logger.info(f"✓ Successfully processed {sample_count} samples")
            
    logger.info("\n✓ Video dataset test completed!")


def test_multi_aspect_ratio_dataset(s3_bucket: str = None, s3_profile: str = None, s3_endpoint_url: str = None):
    """Test multi-aspect ratio dataset with ShardlistMultiAspectRatio distributor in multi-worker scenario"""
    
    logger.info("=== Multi-Aspect Ratio Dataset Test ===")
    
    # Create multi-aspect ratio test datasets
    data_root, dataset_infos = create_multi_aspect_ratio_video_dataset("examples/multi_aspect_data", s3_bucket=s3_bucket, s3_profile=s3_profile, s3_endpoint_url=s3_endpoint_url)
    
    # Create multi-aspect ratio distributor with chunk_size
    distributor = ShardlistMultiAspectRatio(
        shuffle=True,
        split_by_node=True,   # Required to be True
        split_by_worker=True, # Required to be True
        chunk_size=100,
        resume_flag=True,
        verbose=True,
        is_infinite_loader=False,
    )
    
    # Create video decoder
    video_decoder = construct_video_decoder("video_naive_bytes")
    logger.info(f"Created decoder: {video_decoder}")
    
    # Create config with multi-aspect ratio support
    # Create config with video decoder and proper error handler
    config = DatasetConfig(
        keys=[],
        buffer_size=100,
        streaming_download=True,
        dataset_info=dataset_infos,
        distributor=distributor,
        decoders=[video_naive_bytes()],
        augmentation=get_video_augmentor_v2(resolution="512", num_video_frames=60), 
        # shorter video (i.e. the caption window) will be padded to num_video_frames
        # videos shorter than min_duration or fps does not fall in the range will be skipped
        remove_extension_from_keys=True,
        sample_keys_full_list_path=None,
    )

    dataset = Dataset(config=config, decoder_handler=warn_and_continue)
    
    logger.info(f"✓ Built multi-aspect ratio dataset")
    logger.info(f"  Total datasets: {len(dataset_infos)}")
    logger.info(f"  Aspect ratios: {[info.source for info in dataset_infos]}")
    
    # Test with multiple workers
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,  # Multi-worker scenario
        collate_fn=lambda x: x,  # No collation, just return samples as-is
        persistent_workers=False
    )
    
    logger.info("Testing with multi-worker DataLoader...")
    
    # Test loading samples
    sample_count = 0
    aspect_ratios_seen = set()
    
    for batch in dataloader:
        for sample in batch:
            logger.info(f"  Keys: {list(sample.keys())}")
            
            metadata = sample['metas']
            
            aspect_ratio = metadata.get('aspect_ratio', 'unknown')
            width = metadata.get('width', 'unknown')
            height = metadata.get('height', 'unknown')
            logger.info(f" video size: {sample['video'].shape}")
            logger.info(f"  Aspect ratio: {aspect_ratio}")
            logger.info(f"  Dimensions: {width}x{height}")
            aspect_ratios_seen.add(aspect_ratio)
            
            sample_count += 1
    
    logger.info(f"✓ Successfully processed {sample_count} samples")
    logger.info("\n✓ Multi-aspect ratio dataset test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    cosmos_datasets.configure_s3_defaults(
        profile_name="wasabi",
        endpoint_url="https://s3.wasabisys.com", # None for default aws
        region_name="us-east-1"
    )
    
    # Test single aspect ratio first
    test_video_cosmos_dataset(s3_bucket="test-bucket-img2dataset", s3_profile="wasabi", s3_endpoint_url="https://s3.wasabisys.com")
    
    # Test multi-aspect ratio
    test_multi_aspect_ratio_dataset(s3_bucket="test-bucket-img2dataset", s3_profile="wasabi", s3_endpoint_url="https://s3.wasabisys.com")