#!/usr/bin/env python3
"""
Common dataset creation utilities for cosmos_datasets testing.

This module contains reusable functions for creating test datasets
that can be used across different test scripts.
Supports both local and S3 dataset creation.
"""

from cosmos_datasets.webdataset_base.config.schema import DatasetInfo
import json
import logging
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Tuple, List, Optional, Union
import os

from PIL import Image

logger = logging.getLogger(__name__)


def calculate_aspect_ratio(width: int, height: int) -> str:
    """Calculate the closest standard aspect ratio."""
    ratio = width / height
    
    # Define standard aspect ratios and their values
    standard_ratios = {
        "1:1": 1.0,
        "4:3": 4/3,
        "3:4": 3/4,
        "16:9": 16/9,
        "9:16": 9/16
    }
    
    # Find the closest aspect ratio
    closest_ratio = min(standard_ratios.items(), key=lambda x: abs(x[1] - ratio))
    return closest_ratio[0]


def create_test_image(output_path: Path, width: int = 256, height: int = 256, color: tuple = (255, 0, 0)) -> int:
    """
    Create a test image file.
    
    Args:
        output_path: Path where to save the image
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple
        
    Returns:
        File size in bytes
    """
    image = Image.new('RGB', (width, height), color)
    
    # Add some pattern to make it more interesting
    import numpy as np
    pixels = np.array(image)
    
    # Add some noise/pattern
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            end_i = min(i + 10, height)
            end_j = min(j + 10, width)
            pixels[i:end_i, j:end_j] = [
                (color[0] + i) % 255,
                (color[1] + j) % 255,
                (color[2] + i + j) % 255
            ]
    
    image = Image.fromarray(pixels)
    logger.info(f"Creating test image at {output_path} with size {image.size}")
    image.save(output_path, 'JPEG', quality=90)
    return output_path.stat().st_size


def create_blank_video_with_ffmpeg(output_path: Path, duration: float = 2.0, width: int = 320, height: int = 240, fps: int = 24) -> int:
    """
    Create a blank test video using ffmpeg.
    
    Args:
        output_path: Path where to save the video
        duration: Video duration in seconds
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        
    Returns:
        File size in bytes
    """
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create test video with colored background and moving pattern
    cmd = [
        'ffmpeg', '-y',  # Overwrite output file
        '-f', 'lavfi',
        '-i', f'testsrc=duration={duration}:size={width}x{height}:rate={fps}',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Faster encoding
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path.stat().st_size


def upload_to_s3(local_path: Path, bucket_name: str, s3_key: str, 
                 profile_name: str = "default", 
                 endpoint_url: Optional[str] = None) -> None:
    """
    Upload a file to S3.
    
    Args:
        local_path: Local file path to upload
        bucket_name: S3 bucket name
        s3_key: S3 object key
        profile_name: AWS profile name
        endpoint_url: S3 endpoint URL
    """
    try:
        import boto3
        session = boto3.Session(profile_name=profile_name)
        s3_client = session.client('s3', endpoint_url=endpoint_url, region_name='us-east-1')
        
        s3_client.upload_file(str(local_path), bucket_name, s3_key)
        logger.info(f"✓ Uploaded {local_path.name} to s3://{bucket_name}/{s3_key}")
    except ImportError:
        raise ImportError("boto3 is required for S3 upload. Install with: pip install boto3")
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        raise


def create_image_dataset_with_real_images(
    data_save_path: str = "examples/image_test_data", 
    width: int = 512, 
    height: int = 512,
    # S3 options - simplified
    s3_bucket: Optional[str] = None,
    s3_profile: str = "default",
    s3_endpoint_url: Optional[str] = None,
) -> Tuple[Path, DatasetInfo]:
    """
    Create a test image dataset with real image files.
    Supports both local and S3 storage with simplified configuration.
    
    Args:
        data_save_path: Root directory for the dataset (also used as S3 prefix)
        width: Image width in pixels
        height: Image height in pixels
        s3_bucket: S3 bucket name to upload to (if None, creates local dataset only)
        s3_profile: AWS profile name for S3 operations
        s3_endpoint_url: S3 endpoint URL (for non-AWS S3 services)
        
    Returns:
        Tuple of (wdinfo_path, dataset_info)
    """
    data_root = Path(data_save_path)
    data_root.mkdir(parents=True, exist_ok=True)

    # Use data_save_path as-is for S3 prefix
    s3_prefix = data_save_path
    
    # Create directory structure
    image_dir = data_root / "images" / "part_000"
    meta_dir = data_root / "metas" / "part_000"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tar files
    image_tar_path = image_dir / "000.tar"
    metadata_tar_path = meta_dir / "000.tar"
    
    # Create temp directory for images
    temp_dir = data_root / "temp_images"
    temp_dir.mkdir(exist_ok=True)

    num_images = 4  # Number of test images to create
    
    # Create image tar
    with tarfile.open(image_tar_path, "w") as img_tar:
        for i in range(num_images): # Using 4 images for testing
            # Create test image with different colors
            color = (
                (i * 30) % 255,
                (i * 50) % 255,
                (200 - i * 20) % 255
            )
            
            img_path = temp_dir / f"temp_image_{i:03d}.jpg"
            file_size = create_test_image(img_path, width, height, color)

            # Add to tar with key name
            img_tar.add(img_path, arcname=f"{i:03d}.jpg")
    
    # Create metadata tar
    with tarfile.open(metadata_tar_path, "w") as meta_tar:
        for i in range(num_images): # Using 4 images for testing
            # Calculate aspect ratio
            aspect_ratio = calculate_aspect_ratio(width, height)
            
            # Create metadata
            metadata = {
                "sample_id": f"{i:03d}",
                "caption": f"Test image {i} for cosmos_datasets",
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "channels": 3,
                "format": "JPEG",
                "type": "image"
            }
            
            # Save metadata to temp file
            meta_path = temp_dir / f"temp_meta_{i:03d}.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            
            # Add to tar
            meta_tar.add(meta_path, arcname=f"{i:03d}.json")

    # Create wdinfo.json
    # For S3 datasets, use s3:// URLs
    wdinfo_data = {
        "data_keys": ["images", "metas"],
        "chunk_size": 100,
        "data_list": ["part_000/000.tar"],
        "root": f"s3://{s3_bucket}/{s3_prefix}/" if s3_bucket else str(data_root),
        "total_key_count": num_images
    }
    
    wdinfo_path = data_root / "wdinfo.json"
    with open(wdinfo_path, "w") as f:
        json.dump(wdinfo_data, f, indent=2)

    # Upload to S3 if requested
    if s3_bucket:
        logger.info(f"Uploading dataset to S3 bucket: {s3_bucket}")
        
        # Upload tar files
        upload_to_s3(
            image_tar_path, 
            s3_bucket, 
            f"{s3_prefix}/images/part_000/000.tar",
            s3_profile, 
            s3_endpoint_url
        )
        upload_to_s3(
            metadata_tar_path, 
            s3_bucket, 
            f"{s3_prefix}/metas/part_000/000.tar",
            s3_profile, 
            s3_endpoint_url
        )
        
        logger.info(f"✓ Dataset uploaded to s3://{s3_bucket}/{s3_prefix}/")

    # Create dataset info
    dataset_info = [DatasetInfo(
        wdinfo=[str(wdinfo_path)],
        source="image_test",
        opts={"aspect_ratio": '1:1'},  # Required for augmentor
        per_dataset_keys=["jpg", "json"]
    )]
    
    logger.info(f"Created image dataset: {image_tar_path}")
    logger.info(f"Created metadata dataset: {metadata_tar_path}")
    logger.info(f"Created wdinfo: {wdinfo_path}")

    return wdinfo_path, dataset_info


def create_video_dataset_with_real_videos(
    data_save_path: str = "examples/video_test_data", 
    width: int = 512, 
    height: int = 512, 
    duration: float = 4.0,
    # S3 options - simplified
    s3_bucket: Optional[str] = None,
    s3_profile: str = "default",
    s3_endpoint_url: Optional[str] = None
) -> Tuple[Path, DatasetInfo]:
    """
    Create a test video dataset with real video files.
    Supports both local and S3 storage with simplified configuration.
    
    Args:
        data_save_path: Root directory for the dataset (also used as S3 prefix)
        width: Video width in pixels
        height: Video height in pixels  
        duration: Video duration in seconds
        s3_bucket: S3 bucket name to upload to (if None, creates local dataset only)
        s3_profile: AWS profile name for S3 operations
        s3_endpoint_url: S3 endpoint URL (for non-AWS S3 services)
        
    Returns:
        Tuple of (wdinfo_path, dataset_info)
    """
    data_root = Path(data_save_path)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Use data_save_path as-is for S3 prefix
    s3_prefix = data_save_path
    
    # Create directory structure
    video_dir = data_root / "video" / "part_000"
    meta_dir = data_root / "metas" / "part_000"
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tar files
    video_tar_path = video_dir / "000.tar"
    metadata_tar_path = meta_dir / "000.tar"
    
    # Create temp directory for videos
    temp_dir = data_root / "temp_videos"
    temp_dir.mkdir(exist_ok=True)

    num_videos = 4  # Number of test videos to create
    
    # Create video tar
    with tarfile.open(video_tar_path, "w") as vid_tar:
        for i in range(num_videos):
            vid_path = temp_dir / f"temp_video_{i:03d}.mp4"
            file_size = create_blank_video_with_ffmpeg(vid_path, duration, width, height)
            
            # Add to tar with key name
            vid_tar.add(vid_path, arcname=f"{i:03d}.mp4")
    
    # Create metadata tar
    with tarfile.open(metadata_tar_path, "w") as meta_tar:
        for i in range(num_videos):
            # Calculate aspect ratio
            aspect_ratio = calculate_aspect_ratio(width, height)
            
            # Create metadata with proper video parsing structure
            total_frames = int(duration * 24)  # 24 fps
            metadata = {
                "sample_id": f"{i:03d}",
                "caption": f"Test video {i} for cosmos_datasets",
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "fps": 10,
                "format": "MP4",
                "type": "video",
                # Required fields for VideoParsing augmentor
                "framerate": 10,
                "nb_frames": total_frames,
                # Caption windows with start/end frames - single window covering entire video
                "t2w_windows": [
                    {
                        "caption": f"Test video {i} for cosmos_datasets - complete video",
                        "start_frame": 0,
                        "end_frame": 50 # has to be larger than fps*min_duration
                    }
                ]
            }
            
            # Save metadata to temp file
            meta_path = temp_dir / f"temp_meta_{i:03d}.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            
            # Add to tar
            meta_tar.add(meta_path, arcname=f"{i:03d}.json")
    
    
    # Create wdinfo.json
    # For S3 datasets, use s3:// URLs
    wdinfo_data = {
        "data_keys": ["video", "metas"],
        "chunk_size": 8,
        "sequence_length": 4,  # Must be <= chunk_size
        "data_list": ["part_000/000.tar"],
        "root": f"s3://{s3_bucket}/{s3_prefix}/" if s3_bucket else str(data_root),
        "total_key_count": num_videos
    }
    
    wdinfo_path = data_root / "wdinfo.json"
    with open(wdinfo_path, "w") as f:
        json.dump(wdinfo_data, f, indent=2)

    # Upload to S3 if requested
    if s3_bucket:
        logger.info(f"Uploading video dataset to S3 bucket: {s3_bucket}")
        
        # Upload tar files
        upload_to_s3(
            video_tar_path, 
            s3_bucket, 
            f"{s3_prefix}/video/part_000/000.tar",
            s3_profile, 
            s3_endpoint_url
        )
        upload_to_s3(
            metadata_tar_path, 
            s3_bucket, 
            f"{s3_prefix}/metas/part_000/000.tar",
            s3_profile, 
            s3_endpoint_url
        )
        
        logger.info(f"✓ Dataset uploaded to s3://{s3_bucket}/{s3_prefix}/")

    dataset_info = [DatasetInfo(
        wdinfo=[str(wdinfo_path)],
        source="video_test",
        opts={"aspect_ratio": '1:1'},  # Required for augmentor
        per_dataset_keys=["mp4", "json"]  # File extensions we want to load from the tar files
    )]
    
    logger.info(f"Created video dataset: {video_tar_path}")
    logger.info(f"Created metadata dataset: {metadata_tar_path}")
    logger.info(f"Created wdinfo: {wdinfo_path}")

    return wdinfo_path, dataset_info


def create_multi_aspect_ratio_image_dataset(data_save_path: str = "examples/multi_aspect_image_data", s3_bucket: str = None, s3_profile: str = None, s3_endpoint_url: str = None) -> Tuple[Path, List[DatasetInfo]]:
    """
    Create multi-aspect ratio image datasets by leveraging create_image_dataset_with_real_images.
    
    Args:
        data_save_path: Root directory for the datasets
        
    Returns:
        Tuple of (data_root, list_info)
    """
    data_root = Path(data_save_path)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Define aspect ratios and their corresponding dimensions
    aspect_ratios = {
        "1:1": (512, 512),    # 1:1 square
        "16:9": (512, 288),   # 16:9 widescreen
        "4:3": (512, 384)     # 4:3 standard
    }
    
    list_info = []
    for aspect_name, (width, height) in aspect_ratios.items():
        logger.info(f"Creating {aspect_name} image dataset...")
        
        # Create aspect ratio specific directory structure
        aspect_root = data_root / aspect_name
        data_dir = aspect_root / "data"
        
        # Use the existing function with custom image configurations
        wdinfo_path, dataset_info = create_image_dataset_with_real_images(
            data_save_path=str(data_dir),
            width=width,
            height=height,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            s3_endpoint_url=s3_endpoint_url
        )
        list_info.extend(dataset_info)
        
    logger.info(f"created multi-aspect ratio image datasets under {data_root}")
    
    return data_root, list_info


def create_multi_aspect_ratio_video_dataset(data_save_path: str = "examples/multi_aspect_video_data", s3_bucket: str = None, s3_profile: str = None, s3_endpoint_url: str = None) -> Tuple[Path, List[str]]:
    """
    Create multi-aspect ratio video datasets by leveraging create_video_dataset_with_real_videos.
    
    Args:
        data_save_path: Root directory for the datasets
        
    Returns:
        Tuple of (data_root, aspect_ratios)
    """
    data_root = Path(data_save_path)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Define aspect ratios and their corresponding dimensions
    aspect_ratios = {
        "aspect_ratio_1_1": (320, 320),    # 1:1 square
        "aspect_ratio_16_9": (480, 270),   # 16:9 widescreen
        "aspect_ratio_4_3": (320, 240)     # 4:3 standard
    }
    list_info = [] 
    for aspect_name, (width, height) in aspect_ratios.items():
        
        # Create aspect ratio specific directory structure
        aspect_root = data_root / aspect_name
        data_dir = aspect_root / "data"
        wdinfo_dir = aspect_root / "wdinfo"
        wdinfo_dir.mkdir(parents=True, exist_ok=True)
        
        # Create video configs for this specific aspect ratio with shorter durations
        video_configs = [(width, height, 1.0 + (i * 0.5)) for i in range(5)]
        
        # Use the existing function with custom video configurations
        wdinfo_path, dataset_info = create_video_dataset_with_real_videos(
            data_save_path=str(data_dir),
            height=height,
            width=width,
            duration=4.0,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            s3_endpoint_url=s3_endpoint_url
        )
        list_info.extend(dataset_info)

    logger.info(f"created multi-aspect ratio video datasets under {data_root}")

    return data_root, list_info
