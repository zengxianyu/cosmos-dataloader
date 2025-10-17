#!/usr/bin/env python3

"""
Test cosmos_datasets webdataset with actual image files.
This creates proper image files and tests image loading with different decoders.
"""

import sys
from pathlib import Path
import logging
from PIL import Image
import numpy as np

import cosmos_datasets
from cosmos_datasets.log import get_logger
from cosmos_datasets.webdataset import Dataset
from cosmos_datasets.webdataset_base.dataloader import DataLoader
from cosmos_datasets.webdataset_base.config.schema import DatasetConfig
from cosmos_datasets.webdataset_base.distributors.basic import ShardlistBasic
from cosmos_datasets.webdataset_base.distributors.multi_aspect_ratio import ShardlistMultiAspectRatio
import cosmos_datasets.webdataset_base.decoders.image as image_decoders
from cosmos_datasets.dataset_utils import create_image_dataset_with_real_images, create_multi_aspect_ratio_image_dataset
from webdataset.handlers import warn_and_continue
from cosmos_datasets.augmentor_provider import get_image_augmentor

logger = get_logger(__name__)

def get_example_image_dataset(is_infinite: bool = False, s3_bucket: str = None, s3_profile: str = None, s3_endpoint_url: str = None):
    # Create test data with images
    wdinfo_path, dataset_info = create_image_dataset_with_real_images("examples/image_test_data", 
                                                                     s3_bucket=s3_bucket,
                                                                     s3_profile=s3_profile,
                                                                     s3_endpoint_url=s3_endpoint_url)
    
    # Create distributor
    distributor = ShardlistBasic(
        shuffle=True,
        split_by_node=True,
        split_by_worker=True,
        resume_flag=True,
        verbose=True,
        is_infinite_loader=is_infinite,
    )
    
    decoder = image_decoders.pil_loader
    logger.info(f"Using decoder: {decoder}")
    
    # Create config with image decoder
    config = DatasetConfig(
        keys=[], # use key in wdinfo instead
        buffer_size=25,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[decoder],
        augmentation=get_image_augmentor(resolution="512"),
        remove_extension_from_keys=True
    )
    
    # detshuffle: reproducible shuffling mechanism based on a seed and epoch. 
    dataset = Dataset(config=config, decoder_handler=warn_and_continue, detshuffle=True)
    return dataset

def test_image_cosmos_dataset(s3_bucket: str = None, s3_profile: str = None, s3_endpoint_url: str = None):
    """Test cosmos_datasets with actual image data"""
    
    logger.info("=== Image cosmos_datasets Test ===")

    built_dataset = get_example_image_dataset(s3_bucket=s3_bucket, s3_profile=s3_profile, s3_endpoint_url=s3_endpoint_url).build_dataset()
    logger.info(f"✓ Built dataset with pil_loader decoder")
    logger.info(f"  Total samples: {getattr(built_dataset, 'total_images', 'N/A')}")
    
    # Test loading samples
    sample_count = 0
    
    for sample in built_dataset:
        logger.info(f"  Keys: {list(sample.keys())}")
        
        # Check image data
        logger.info(f"  Image size: {sample['images'].shape}")
        logger.info(f"  Image mode: {sample['images'].dtype}")
        
        # Check metadata
        metadata = sample['metas']
        logger.info(f"Sample metas: {metadata}")
        
        sample_count += 1
    
    logger.info("\n✓ Image dataset test completed!")


def test_multi_aspect_ratio_image_dataset(s3_bucket: str = None, s3_profile: str = None, s3_endpoint_url: str = None):
    """Test multi-aspect ratio image dataset with ShardlistMultiAspectRatio distributor"""
    
    logger.info("=== Multi-Aspect Ratio Image Dataset Test ===")
    
    # Create multi-aspect ratio test datasets
    data_root, dataset_infos = create_multi_aspect_ratio_image_dataset("examples/multi_aspect_image_data", s3_bucket=s3_bucket, s3_profile=s3_profile, s3_endpoint_url=s3_endpoint_url)
    
    # Create multi-aspect ratio distributor
    distributor = ShardlistMultiAspectRatio(
        shuffle=True,
        split_by_node=True,
        split_by_worker=True,
        chunk_size=10,
        resume_flag=True,
        verbose=True
    )
    
    # Create image decoder
    decoder = image_decoders.pil_loader
    logger.info(f"Using decoder: {decoder}")
    
    # Create config with multi-aspect ratio support
    config = DatasetConfig(
        keys=[],
        buffer_size=25,
        streaming_download=True,
        dataset_info=dataset_infos,
        distributor=distributor,
        decoders=[decoder],
        augmentation=get_image_augmentor(resolution="512"),
        remove_extension_from_keys=True
    )

    dataset = Dataset(config=config, decoder_handler=warn_and_continue)
    
    logger.info(f"✓ Built multi-aspect ratio image dataset")
    logger.info(f"  Total datasets: {len(dataset_infos)}")
    logger.info(f"  Aspect ratios: {[info.source for info in dataset_infos]}")
    
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        collate_fn=lambda x: x,
        persistent_workers=False
    )
    
    logger.info("Testing with multi-worker DataLoader...")
    
    # Test loading samples
    sample_count = 0
    aspect_ratios_seen = set()
    
    for batch in dataloader:
        for sample in batch:
            logger.info(f"  Keys: {list(sample.keys())}")
            
            # Check image data
            logger.info(f"  Image size: {sample['images'].shape}")
            logger.info(f"  Image dtype: {sample['images'].dtype}")

            # Check metadata for aspect ratio info
            metadata = sample['metas']
                    
            aspect_ratio = metadata.get('aspect_ratio', 'unknown')
            width = metadata.get('width', 'unknown')
            height = metadata.get('height', 'unknown')
            logger.info(f"  Aspect ratio: {aspect_ratio}")
            logger.info(f"  Dimensions: {width}x{height}")
            aspect_ratios_seen.add(aspect_ratio)
            
            sample_count += 1
                
    logger.info("\n✓ Multi-aspect ratio image dataset test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    # Test single aspect ratio first
    test_image_cosmos_dataset()
    
    # Test multi-aspect ratio
    test_multi_aspect_ratio_image_dataset()

    # for s3
    ## Configure S3 defaults
    #cosmos_datasets.configure_s3_defaults(
    #    profile_name="wasabi",
    #    endpoint_url="https://s3.wasabisys.com", # None for default aws
    #    region_name="us-east-1"
    #)
    
    ## Test single aspect ratio first
    #test_image_cosmos_dataset(s3_bucket="test-bucket-img2dataset", s3_profile="wasabi", s3_endpoint_url="https://s3.wasabisys.com")
    
    ## Test multi-aspect ratio
    #test_multi_aspect_ratio_image_dataset(s3_bucket="test-bucket-img2dataset", s3_profile="wasabi", s3_endpoint_url="https://s3.wasabisys.com")