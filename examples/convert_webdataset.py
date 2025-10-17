#!/usr/bin/env python3
"""
Conversion script to convert conventional webdataset tar files to cosmos-datasets format.

Conventional webdataset format:
- All modalities (e.g., .jpg, .txt, .json) are in the same tar file
- Files are named like: 000000.jpg, 000000.txt, 000001.jpg, 000001.txt, etc.

Cosmos-datasets format:
- Different modalities are in separate tar files under different directories
- Images go to: images/part_000/000.tar
- Metadata goes to: metas/part_000/000.tar
- Each tar contains files with same base name but different extensions
"""

import argparse
import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import tempfile
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from multiprocessing import Pool, cpu_count
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_s3_client(aws_profile: str = None, endpoint_url: str = None):
    """Create and return S3 client with optional profile and endpoint."""
    try:
        if aws_profile and aws_profile != "default":
            session = boto3.Session(profile_name=aws_profile)
            return session.client('s3', endpoint_url=endpoint_url)
        else:
            return boto3.client('s3', endpoint_url=endpoint_url)
    except (NoCredentialsError, Exception) as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise


def extract_sample_info_from_s3(s3_client, bucket: str, key: str) -> Tuple[Dict[str, Set[str]], int]:
    """
    Extract information about samples in a conventional webdataset tar file from S3.
    
    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key (path to tar file)
        
    Returns:
        Tuple of (sample_extensions_map, total_samples)
    """
    sample_extensions = {}
    
    try:
        # Stream the tar file from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        tar_data = response['Body'].read()
        
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    name = member.name
                    # Remove any directory path and get just the filename
                    filename = os.path.basename(name)
                    
                    # Split filename to get base name and extension
                    if '.' in filename:
                        base_name, ext = filename.rsplit('.', 1)
                        
                        if base_name not in sample_extensions:
                            sample_extensions[base_name] = set()
                        sample_extensions[base_name].add(ext)
        
        return sample_extensions, len(sample_extensions)
    
    except ClientError as e:
        logger.error(f"Error reading S3 object s3://{bucket}/{key}: {e}")
        raise


def extract_sample_info(tar_path: Path) -> Tuple[Dict[str, Set[str]], int]:
    """
    Extract information about samples in a conventional webdataset tar file.
    
    Args:
        tar_path: Path to the conventional webdataset tar file
        
    Returns:
        Tuple of (sample_extensions_map, total_samples)
        sample_extensions_map: Dict mapping sample IDs to set of extensions
        total_samples: Total number of unique samples
    """
    sample_extensions = {}
    
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                name = member.name
                # Remove any directory path and get just the filename
                filename = os.path.basename(name)
                
                # Split filename to get base name and extension
                if '.' in filename:
                    base_name, ext = filename.rsplit('.', 1)
                    
                    if base_name not in sample_extensions:
                        sample_extensions[base_name] = set()
                    sample_extensions[base_name].add(ext)
    
    return sample_extensions, len(sample_extensions)


def categorize_extensions(sample_extensions: Dict[str, Set[str]], extension_mapping: Dict[str, str] = None) -> Dict[str, Set[str]]:
    """
    Categorize extensions into different folders based on file type or custom mapping.
    
    Args:
        sample_extensions: Dict mapping sample IDs to set of extensions
        extension_mapping: Optional dict mapping extensions to folder names
        
    Returns:
        Dict mapping folder names to sets of extensions
    """
    # Get all unique extensions
    all_extensions = set()
    for extensions in sample_extensions.values():
        all_extensions.update(extensions)
    
    # Define default categorization
    image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}
    
    categorized_extensions = {}
    
    for ext in all_extensions:
        ext_lower = ext.lower()
        
        # Check if there's a custom mapping for this extension
        if extension_mapping and ext in extension_mapping:
            folder_name = extension_mapping[ext]
        elif extension_mapping and ext_lower in extension_mapping:
            folder_name = extension_mapping[ext_lower]
        # Default categorization
        elif ext_lower in image_extensions:
            folder_name = "images"
        elif ext_lower in video_extensions:
            folder_name = "video"
        else:
            # Skip unknown extensions by default
            logger.warning(f"Unknown extension '{ext}', skipping (use --extension-mapping to include)")
            continue
        
        if folder_name not in categorized_extensions:
            categorized_extensions[folder_name] = set()
        categorized_extensions[folder_name].add(ext)
    
    return categorized_extensions


def convert_webdataset_tar_from_s3(
    s3_client,
    input_bucket: str,
    input_key: str,
    output_bucket: str,
    output_prefix: str,
    extension_mapping: Dict[str, str] = None,
    part_name: str = "part_000",
    output_tar_name: str = "000.tar"
) -> Tuple[Dict[str, str], Dict]:
    """
    Convert a conventional webdataset tar file from S3 to cosmos-datasets format in S3.
    
    Args:
        s3_client: boto3 S3 client
        input_bucket: Input S3 bucket name
        input_key: Input S3 object key (path to tar file)
        output_bucket: Output S3 bucket name
        output_prefix: Output S3 prefix
        extension_mapping: Dict mapping extensions to folder names
        part_name: Part directory name (e.g., "part_000")
        output_tar_name: Output tar file name (e.g., "000.tar")
        
    Returns:
        Tuple of (folder_s3_keys, conversion_info)
        folder_s3_keys: Dict mapping folder names to S3 keys
        conversion_info: Dict with conversion statistics
    """
    # Extract sample information
    sample_extensions, total_samples = extract_sample_info_from_s3(s3_client, input_bucket, input_key)
    categorized_extensions = categorize_extensions(sample_extensions, extension_mapping)
    
    logger.info(f"Found {total_samples} samples in s3://{input_bucket}/{input_key}")
    for folder_name, extensions in categorized_extensions.items():
        logger.info(f"{folder_name} extensions: {sorted(extensions)}")
    
    # Create output S3 keys
    folder_s3_keys = {}
    file_counts = {}
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download and extract the original tar file
        logger.info(f"Downloading s3://{input_bucket}/{input_key}...")
        response = s3_client.get_object(Bucket=input_bucket, Key=input_key)
        tar_data = response['Body'].read()
        
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as input_tar:
            input_tar.extractall(temp_path)
        
        # Create tar files for each folder
        for folder_name, extensions in categorized_extensions.items():
            # Create S3 key for this folder's tar file
            clean_output_prefix = output_prefix.rstrip('/') if output_prefix else ""
            if part_name:
                # Traditional cosmos format with part directories
                if clean_output_prefix:
                    s3_key = f"{clean_output_prefix}/{folder_name}/{part_name}/{output_tar_name}"
                else:
                    s3_key = f"{folder_name}/{part_name}/{output_tar_name}"
            else:
                # Simplified format with just folder/filename.tar
                if clean_output_prefix:
                    s3_key = f"{clean_output_prefix}/{folder_name}/{output_tar_name}"
                else:
                    s3_key = f"{folder_name}/{output_tar_name}"
            folder_s3_keys[folder_name] = s3_key
            
            # Create tar file for this folder in memory
            tar_buffer = io.BytesIO()
            file_count = 0
            
            with tarfile.open(fileobj=tar_buffer, mode="w") as folder_tar:
                for sample_id in sorted(sample_extensions.keys()):
                    sample_exts = sample_extensions[sample_id]
                    
                    # Add files with extensions belonging to this folder
                    for ext in sample_exts:
                        if ext in extensions:
                            original_filename = f"{sample_id}.{ext}"
                            file_path = temp_path / original_filename
                            
                            if file_path.exists():
                                # Add to folder tar with clean filename
                                folder_tar.add(file_path, arcname=original_filename)
                                file_count += 1
            
            # Upload tar file to S3
            tar_buffer.seek(0)
            logger.info(f"Uploading {folder_name} tar to s3://{output_bucket}/{s3_key}")
            s3_client.put_object(
                Bucket=output_bucket,
                Key=s3_key,
                Body=tar_buffer.getvalue()
            )
            
            file_counts[folder_name] = file_count
            logger.info(f"✓ Uploaded {folder_name} tar: s3://{output_bucket}/{s3_key} ({file_count} files)")
    
    conversion_info = {
        "total_samples": total_samples,
        "file_counts": file_counts,
        "categorized_extensions": categorized_extensions
    }
    
    return folder_s3_keys, conversion_info


def convert_webdataset_tar(
    input_tar_path: Path,
    output_dir: Path,
    extension_mapping: Dict[str, str] = None,
    part_name: str = "part_000",
    output_tar_name: str = "000.tar"
) -> Tuple[Dict[str, Path], Dict]:
    """
    Convert a conventional webdataset tar file to cosmos-datasets format.
    
    Args:
        input_tar_path: Path to input conventional webdataset tar file
        output_dir: Output directory for converted dataset
        extension_mapping: Dict mapping extensions to folder names
        part_name: Part directory name (e.g., "part_000")
        output_tar_name: Output tar file name (e.g., "000.tar")
        
    Returns:
        Tuple of (folder_tar_paths, conversion_info)
        folder_tar_paths: Dict mapping folder names to tar file paths
        conversion_info: Dict with conversion statistics
    """
    # Extract sample information
    sample_extensions, total_samples = extract_sample_info(input_tar_path)
    categorized_extensions = categorize_extensions(sample_extensions, extension_mapping)
    
    logger.info(f"Found {total_samples} samples in {input_tar_path}")
    for folder_name, extensions in categorized_extensions.items():
        logger.info(f"{folder_name} extensions: {sorted(extensions)}")
    
    # Create output directory structure and tar files
    folder_tar_paths = {}
    file_counts = {}
    
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract the original tar file
        with tarfile.open(input_tar_path, "r") as input_tar:
            input_tar.extractall(temp_path)
        
        # Create tar files for each folder
        for folder_name, extensions in categorized_extensions.items():
            # Create directory structure
            folder_dir = output_dir / folder_name / part_name
            folder_dir.mkdir(parents=True, exist_ok=True)
            
            folder_tar_path = folder_dir / output_tar_name
            folder_tar_paths[folder_name] = folder_tar_path
            
            # Create tar file for this folder
            file_count = 0
            with tarfile.open(folder_tar_path, "w") as folder_tar:
                for sample_id in sorted(sample_extensions.keys()):
                    sample_exts = sample_extensions[sample_id]
                    
                    # Add files with extensions belonging to this folder
                    for ext in sample_exts:
                        if ext in extensions:
                            original_filename = f"{sample_id}.{ext}"
                            
                            # Find the file in the extracted directory
                            file_path = None
                            for root, dirs, files in os.walk(temp_path):
                                if original_filename in files:
                                    file_path = Path(root) / original_filename
                                    break
                            
                            if file_path and file_path.exists():
                                # Add to folder tar with clean filename
                                folder_tar.add(file_path, arcname=original_filename)
                                file_count += 1
            
            file_counts[folder_name] = file_count
            logger.info(f"Created {folder_name} tar: {folder_tar_path} ({file_count} files)")
    
    conversion_info = {
        "total_samples": total_samples,
        "file_counts": file_counts,
        "categorized_extensions": categorized_extensions
    }
    
    return folder_tar_paths, conversion_info


def create_wdinfo_file(
    output_dir: Path,
    folder_names: List[str],
    part_name: str,
    output_tar_name: str,
    total_samples: int,
    chunk_size: int = 100,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None
) -> Path:
    """
    Create wdinfo.json file for the converted dataset.
    
    Args:
        output_dir: Output directory for the dataset
        folder_names: List of folder names (data keys)
        part_name: Part directory name
        output_tar_name: Output tar file name
        total_samples: Total number of samples
        chunk_size: Chunk size for the dataset
        s3_bucket: Optional S3 bucket name
        s3_prefix: Optional S3 prefix
        
    Returns:
        Path to the created wdinfo.json file
    """
    # Create wdinfo data
    wdinfo_data = {
        "data_keys": folder_names,
        "chunk_size": chunk_size,
        "data_list": [f"{part_name}/{output_tar_name}"],
        "total_key_count": total_samples
    }
    
    # Set root path
    if s3_bucket and s3_prefix:
        wdinfo_data["root"] = f"s3://{s3_bucket}/{s3_prefix}/"
    elif s3_bucket:
        wdinfo_data["root"] = f"s3://{s3_bucket}/"
    else:
        wdinfo_data["root"] = str(output_dir)
    
    # Write wdinfo.json
    wdinfo_path = output_dir / "wdinfo.json"
    with open(wdinfo_path, "w") as f:
        json.dump(wdinfo_data, f, indent=2)
    
    logger.info(f"Created wdinfo.json: {wdinfo_path}")
    return wdinfo_path


def convert_single_s3_tar_worker(args):
    """
    Worker function for multiprocessing conversion of a single S3 tar file.
    
    Args:
        args: Tuple containing (file_info, s3_config, conversion_config)
            file_info: (file_num, input_bucket, input_key, output_bucket, output_prefix)
            s3_config: (aws_profile, endpoint_url)
            conversion_config: (extension_mapping, output_tar_name)
    
    Returns:
        Tuple: (success, file_num, folder_s3_keys, conversion_info, error_msg, output_tar_name)
    """
    try:
        file_info, s3_config, conversion_config = args
        file_num, input_bucket, input_key, output_bucket, output_prefix = file_info
        aws_profile, endpoint_url = s3_config
        extension_mapping, output_tar_name = conversion_config
        
        # Create S3 client for this worker
        s3_client = get_s3_client(aws_profile, endpoint_url)
        
        logger.info(f"Converting s3://{input_bucket}/{input_key} -> {output_tar_name}")
        
        folder_s3_keys, conversion_info = convert_webdataset_tar_from_s3(
            s3_client,
            input_bucket,
            input_key,
            output_bucket,
            output_prefix,
            extension_mapping,
            "",  # No part_name needed since we're using the original filename
            output_tar_name
        )
        
        return (True, file_num, folder_s3_keys, conversion_info, None, output_tar_name)
        
    except Exception as e:
        return (False, file_num, None, None, str(e), output_tar_name)


def convert_multiple_s3_tars(
    s3_client,
    input_bucket: str,
    input_prefix: str,
    output_bucket: str,
    output_prefix: str,
    extension_mapping: Dict[str, str] = None,
    chunk_size: int = 100,
    file_pattern: str = "{:05d}.tar",
    start_file: int = 0,
    max_files: int = None,
    num_processes: int = None,
    aws_profile: str = None,
    endpoint_url: str = None
) -> str:
    """
    Convert multiple conventional webdataset tar files from S3 to cosmos-datasets format using multiprocessing.
    
    Args:
        s3_client: boto3 S3 client
        input_bucket: Input S3 bucket name
        input_prefix: Input S3 prefix
        output_bucket: Output S3 bucket name
        output_prefix: Output S3 prefix
        extension_mapping: Dict mapping extensions to folder names
        chunk_size: Chunk size for wdinfo
        file_pattern: Pattern for tar file names
        start_file: Starting file number
        max_files: Maximum number of files to process
        num_processes: Number of parallel processes (default: CPU count)
        aws_profile: AWS profile name for worker processes
        endpoint_url: S3 endpoint URL for worker processes
        
    Returns:
        S3 key of the created wdinfo.json file
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    logger.info(f"Using {num_processes} processes for conversion")
    
    # First, discover all available tar files by listing the S3 prefix
    logger.info("Discovering available tar files...")
    
    # Use S3 list_objects_v2 to get all tar files in the prefix
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=input_bucket,
            Prefix=input_prefix.rstrip('/') + '/' if input_prefix else ''
        )
        
        tar_files = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = key.split('/')[-1]  # Get just the filename
                    if filename.endswith('.tar') and filename.startswith('00'):
                        # Extract the number from filename like 00123.tar
                        try:
                            file_num = int(filename[:-4])  # Remove .tar and convert to int
                            tar_files.append((file_num, filename, key))
                        except ValueError:
                            continue
        
        # Sort by file number
        tar_files.sort(key=lambda x: x[0])
        
        # Apply start_file and max_files limits
        filtered_files = [f for f in tar_files if f[0] >= start_file]
        if max_files:
            filtered_files = filtered_files[:max_files]
        
        logger.info(f"Found {len(filtered_files)} tar files to convert (out of {len(tar_files)} total)")
        
        if not filtered_files:
            raise ValueError("No tar files found to convert")
        
        # Prepare file info list
        file_info_list = []
        for file_num, filename, input_key in filtered_files:
            file_info = (file_num, input_bucket, input_key, output_bucket, output_prefix)
            file_info_list.append((file_info, filename))  # Use original filename
            
    except Exception as e:
        logger.error(f"Error discovering files: {e}")
        raise
    
    logger.info(f"Found {len(file_info_list)} tar files to convert")
    
    # Prepare arguments for multiprocessing
    s3_config = (aws_profile, endpoint_url)
    worker_args = []
    
    for file_info, output_tar_name in file_info_list:
        conversion_config = (extension_mapping, output_tar_name)
        worker_args.append((file_info, s3_config, conversion_config))
    
    # Process files in parallel
    total_samples = 0
    data_list = []
    all_folder_names = set()
    successful_conversions = 0
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(convert_single_s3_tar_worker, worker_args)
    
    # Process results
    for success, file_num, folder_s3_keys, conversion_info, error_msg, output_tar_name in results:
        if success:
            total_samples += conversion_info["total_samples"]
            data_list.append(output_tar_name)  # Use original tar name like 00000.tar
            all_folder_names.update(folder_s3_keys.keys())
            successful_conversions += 1
        else:
            logger.error(f"Failed to convert file {file_num}: {error_msg}")
    
    if successful_conversions == 0:
        raise ValueError("No files were successfully converted")
    
    # Create combined wdinfo.json
    folder_names = sorted(all_folder_names)
    
    if output_bucket:
        # S3 output
        clean_output_prefix = output_prefix.rstrip('/') if output_prefix else ""
        wdinfo_data = {
            "data_keys": folder_names,
            "chunk_size": chunk_size,
            "data_list": data_list,
            "total_key_count": total_samples,
            "root": f"s3://{output_bucket}/{clean_output_prefix}/" if clean_output_prefix else f"s3://{output_bucket}/"
        }
        
        # Upload wdinfo.json to S3
        wdinfo_key = f"{clean_output_prefix}/wdinfo.json" if clean_output_prefix else "wdinfo.json"
        wdinfo_content = json.dumps(wdinfo_data, indent=2)
        
        s3_client.put_object(
            Bucket=output_bucket,
            Key=wdinfo_key,
            Body=wdinfo_content.encode('utf-8'),
            ContentType='application/json'
        )
        
        logger.info(f"Conversion complete! Total samples: {total_samples}")
        logger.info(f"Successfully processed: {successful_conversions}/{len(file_info_list)} files")
        logger.info(f"Created wdinfo.json: s3://{output_bucket}/{wdinfo_key}")
        
        return wdinfo_key
    else:
        # Local output
        wdinfo_data = {
            "data_keys": folder_names,
            "chunk_size": chunk_size,
            "data_list": data_list,
            "total_key_count": total_samples,
            "root": str(Path(output_prefix).resolve()) + "/"
        }
        
        # Save wdinfo.json locally
        wdinfo_path = Path(output_prefix) / "wdinfo.json"
        with open(wdinfo_path, 'w') as f:
            json.dump(wdinfo_data, f, indent=2)
        
        logger.info(f"Conversion complete! Total samples: {total_samples}")
        logger.info(f"Successfully processed: {successful_conversions}/{len(file_info_list)} files")
        logger.info(f"Created wdinfo.json: {wdinfo_path}")
        
        return str(wdinfo_path)


def convert_multiple_tars(
    input_dir: Path,
    output_dir: Path,
    extension_mapping: Dict[str, str] = None,
    chunk_size: int = 100,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None
) -> Path:
    """
    Convert multiple conventional webdataset tar files to cosmos-datasets format.
    
    Args:
        input_dir: Directory containing input tar files
        output_dir: Output directory for converted dataset
        extension_mapping: Dict mapping extensions to folder names
        chunk_size: Chunk size for wdinfo
        s3_bucket: Optional S3 bucket name
        s3_prefix: Optional S3 prefix
        
    Returns:
        Path to the created wdinfo.json file
    """
    # Find all tar files in input directory
    tar_files = sorted(list(input_dir.glob("*.tar")))
    if not tar_files:
        raise ValueError(f"No tar files found in {input_dir}")
    
    logger.info(f"Found {len(tar_files)} tar files to convert")
    
    total_samples = 0
    data_list = []
    all_folder_names = set()
    
    # Convert each tar file
    for i, tar_file in enumerate(tar_files):
        part_name = f"part_{i:03d}"
        output_tar_name = "000.tar"
        
        logger.info(f"Converting {tar_file} -> {part_name}")
        
        folder_tar_paths, conversion_info = convert_webdataset_tar(
            tar_file,
            output_dir,
            extension_mapping,
            part_name,
            output_tar_name
        )
        
        total_samples += conversion_info["total_samples"]
        data_list.append(f"{part_name}/{output_tar_name}")
        all_folder_names.update(folder_tar_paths.keys())
    
    # Create combined wdinfo.json
    folder_names = sorted(all_folder_names)
    
    wdinfo_data = {
        "data_keys": folder_names,
        "chunk_size": chunk_size,
        "data_list": data_list,
        "total_key_count": total_samples
    }
    
    # Set root path
    if s3_bucket and s3_prefix:
        wdinfo_data["root"] = f"s3://{s3_bucket}/{s3_prefix}/"
    elif s3_bucket:
        wdinfo_data["root"] = f"s3://{s3_bucket}/"
    else:
        wdinfo_data["root"] = str(output_dir)
    
    # Write wdinfo.json
    wdinfo_path = output_dir / "wdinfo.json"
    with open(wdinfo_path, "w") as f:
        json.dump(wdinfo_data, f, indent=2)
    
    logger.info(f"Conversion complete! Total samples: {total_samples}")
    logger.info(f"Created wdinfo.json: {wdinfo_path}")
    
    return wdinfo_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert conventional webdataset tar files to cosmos-datasets format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input tar file, directory containing tar files, or S3 URI (s3://bucket/prefix/)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for converted dataset or S3 URI (s3://bucket/prefix/)"
    )
    parser.add_argument(
        "--extension-mapping",
        type=str,
        default='{\"txt\":\"captions\",\"json\":\"metas\"}',
        help="JSON string mapping extensions to folder names, e.g., '{\"txt\":\"captions\",\"json\":\"metas\"}'"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size for wdinfo (default: 100)"
    )
    parser.add_argument(
        "--s3-bucket",
        help="S3 bucket name (optional, inferred from S3 URIs)"
    )
    parser.add_argument(
        "--s3-prefix",
        help="S3 prefix (optional, inferred from S3 URIs)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        help="Number of parallel processes for S3 conversion (default: CPU count)"
    )
    parser.add_argument(
        "--aws-profile",
        help="AWS profile name for S3 access"
    )
    parser.add_argument(
        "--s3-endpoint-url",
        help="S3 endpoint URL (for S3-compatible services like Wasabi)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Parse extension mapping if provided
    extension_mapping = None
    if args.extension_mapping:
        try:
            extension_mapping = json.loads(args.extension_mapping)
            logger.info(f"Using extension mapping: {extension_mapping}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in extension-mapping: {e}")
    
    # Check if input is S3 URI
    is_s3_input = args.input_path.startswith("s3://")
    is_s3_output = args.output_dir.startswith("s3://")
    
    if is_s3_input:
        # Handle S3 input
        logger.info(f"Converting from S3: {args.input_path}")
        
        # Create S3 client
        s3_client = get_s3_client(args.aws_profile, args.s3_endpoint_url)
        
        # Parse S3 URI
        s3_parts = args.input_path[5:].split("/", 1)  # Remove s3:// prefix
        input_bucket = s3_parts[0]
        input_prefix = s3_parts[1] if len(s3_parts) > 1 else ""
        
        if is_s3_output:
            # S3 to S3 conversion
            # Clean the output URI first to avoid double slash issues
            clean_output_uri = args.output_dir.rstrip('/')
            output_s3_parts = clean_output_uri[5:].split("/", 1)
            output_bucket = output_s3_parts[0]
            output_prefix = output_s3_parts[1] if len(output_s3_parts) > 1 else ""
            
            wdinfo_path = convert_multiple_s3_tars(
                s3_client,
                input_bucket,
                input_prefix,
                output_bucket,
                output_prefix,
                extension_mapping,
                args.chunk_size,
                "{:05d}.tar",
                0,
                args.max_files,
                args.num_processes,
                args.aws_profile,
                args.s3_endpoint_url
            )
        else:
            # S3 to local conversion
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            wdinfo_path = convert_multiple_s3_tars(
                s3_client,
                input_bucket,
                input_prefix,
                None,  # No output bucket for local
                str(output_path),
                extension_mapping,
                args.chunk_size,
                "{:05d}.tar",
                0,
                args.max_files,
                args.num_processes,
                args.aws_profile,
                args.s3_endpoint_url
            )
    else:
        # Handle local input (existing logic)
        input_path = Path(args.input_path)
        
        if is_s3_output:
            raise ValueError("S3 output with local input is not yet supported")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if input_path.is_file() and input_path.suffix == ".tar":
            # Convert single tar file
            logger.info(f"Converting single tar file: {input_path}")
            
            folder_tar_paths, conversion_info = convert_webdataset_tar(
                input_path,
                output_dir,
                extension_mapping
            )
            
            folder_names = sorted(folder_tar_paths.keys())
            wdinfo_path = create_wdinfo_file(
                output_dir,
                folder_names,
                "part_000",
                "000.tar",
                conversion_info["total_samples"],
                args.chunk_size,
                args.s3_bucket,
                args.s3_prefix
            )
            
        elif input_path.is_dir():
            # Convert multiple tar files
            logger.info(f"Converting tar files from directory: {input_path}")
            
            wdinfo_path = convert_multiple_tars(
                input_path,
                output_dir,
                extension_mapping,
                args.chunk_size,
                args.s3_bucket,
                args.s3_prefix
            )
            
        else:
            raise ValueError(f"Input path must be a tar file or directory: {input_path}")
    
    logger.info("Conversion completed successfully!")
    logger.info(f"Output dataset: {args.output_dir}")
    logger.info(f"WDInfo file: {wdinfo_path}")
    
    # Show final directory structure for local output
    if not is_s3_output:
        output_path = Path(args.output_dir)
        logger.info("Created directory structure:")
        for root, dirs, files in os.walk(output_path):
            root_path = Path(root)
            level = len(root_path.relative_to(output_path).parts)
            indent = "  " * level
            logger.info(f"{indent}{root_path.name}/")
            subindent = "  " * (level + 1)
            for file in files:
                logger.info(f"{subindent}{file}")


if __name__ == "__main__":
    main()