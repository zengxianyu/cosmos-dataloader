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

"""
S3 streaming support for cosmos-datasets WebDataset.
Provides native s3:// URL support without requiring pre-signed URLs or monkey patching.
"""

import os
import logging
from typing import BinaryIO, Optional, Dict, Any
from urllib.parse import urlparse
import threading
# Removed unused import

logger = logging.getLogger(__name__)

# Global S3 client cache and lock
_s3_clients: Dict[str, Any] = {}
_s3_client_lock = threading.Lock()

def get_s3_client(profile_name: Optional[str] = None, 
                  endpoint_url: Optional[str] = None,
                  region_name: Optional[str] = None) -> Any:
    """Get or create an S3 client with caching."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
    except ImportError:
        raise ImportError("boto3 is required for S3 support. Install with: pip install boto3")
    
    # Create cache key
    cache_key = f"{profile_name}:{endpoint_url}:{region_name}"
    
    with _s3_client_lock:
        if cache_key in _s3_clients:
            return _s3_clients[cache_key]
        
        # Create new S3 client
        try:
            if profile_name:
                session = boto3.Session(profile_name=profile_name)
            else:
                session = boto3.Session()
            
            client_kwargs = {}
            if endpoint_url:
                client_kwargs['endpoint_url'] = endpoint_url
            if region_name:
                client_kwargs['region_name'] = region_name
            
            s3_client = session.client('s3', **client_kwargs)
            
            # Test the client
            s3_client.list_buckets()
            
            _s3_clients[cache_key] = s3_client
            logger.debug(f"Created S3 client for profile={profile_name}, endpoint={endpoint_url}")
            
            return s3_client
            
        except Exception as e:
            logger.error(f"Failed to create S3 client: {e}")
            raise

def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse s3://bucket/key URL into bucket and key components."""
    parsed = urlparse(url)
    if parsed.scheme != 's3':
        raise ValueError(f"Not an S3 URL: {url}")
    
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    if not bucket:
        raise ValueError(f"Invalid S3 URL - no bucket specified: {url}")
    
    return bucket, key

class S3StreamWrapper:
    """Wrapper for S3 streaming body to provide file-like interface."""
    
    def __init__(self, s3_response_body):
        self._body = s3_response_body
        self._closed = False
    
    def read(self, size: int = -1) -> bytes:
        """Read data from S3 stream."""
        if self._closed:
            raise ValueError("I/O operation on closed stream")
        return self._body.read(size)
    
    def close(self) -> None:
        """Close the S3 stream."""
        if not self._closed:
            self._body.close()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line
    
    def readline(self, size: int = -1) -> bytes:
        """Read a line from the stream."""
        # Simple implementation - read byte by byte until newline
        line = b''
        while True:
            if size > 0 and len(line) >= size:
                break
            char = self.read(1)
            if not char:
                break
            line += char
            if char == b'\n':
                break
        return line

def gopen_s3(url: str, mode: str = "rb", bufsize: int = 8192, **kwargs) -> BinaryIO:
    """
    Open S3 URLs for streaming.
    
    Args:
        url: S3 URL in format s3://bucket/key
        mode: File mode (only 'rb' supported)
        bufsize: Buffer size (unused for S3)
        **kwargs: Additional S3 configuration:
            - s3_profile: AWS profile name
            - s3_endpoint_url: Custom S3 endpoint (e.g., for Wasabi)
            - s3_region: AWS region name
    
    Returns:
        File-like object for streaming S3 content
    """
    if mode != "rb":
        raise ValueError(f"S3 streaming only supports 'rb' mode, got: {mode}")
    
    # Parse S3 URL
    bucket, key = parse_s3_url(url)
    
    # Get S3 configuration from environment or kwargs
    profile_name = kwargs.get('s3_profile') or os.environ.get('S3_PROFILE')
    endpoint_url = kwargs.get('s3_endpoint_url') or os.environ.get('S3_ENDPOINT_URL')
    region_name = kwargs.get('s3_region') or os.environ.get('S3_REGION', 'us-east-1')
    
    logger.debug(f"Opening S3 stream: s3://{bucket}/{key}")
    
    try:
        # Get S3 client
        s3_client = get_s3_client(profile_name, endpoint_url, region_name)
        
        # Get object
        response = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Return wrapped streaming body
        return S3StreamWrapper(response['Body'])
        
    except Exception as e:
        logger.error(f"Failed to open S3 stream s3://{bucket}/{key}: {e}")
        raise

def configure_s3_defaults(profile_name: Optional[str] = None,
                         endpoint_url: Optional[str] = None,
                         region_name: Optional[str] = None) -> None:
    """
    Configure default S3 settings for the cosmos-datasets package.
    
    Args:
        profile_name: Default AWS profile name
        endpoint_url: Default S3 endpoint URL (e.g., for Wasabi)
        region_name: Default AWS region
    """
    if profile_name:
        os.environ['S3_PROFILE'] = profile_name
    if endpoint_url:
        os.environ['S3_ENDPOINT_URL'] = endpoint_url
    if region_name:
        os.environ['S3_REGION'] = region_name
    
    logger.info(f"Configured S3 defaults: profile={profile_name}, endpoint={endpoint_url}, region={region_name}")

# Auto-register S3 handler with webdataset when this module is imported
def register_s3_handler():
    """Register S3 handler with webdataset gopen system."""
    try:
        import webdataset
        webdataset.gopen_schemes['s3'] = gopen_s3
        logger.info("✓ Registered native S3 streaming support")
    except Exception as e:
        logger.warning(f"Failed to register S3 handler: {e}")

# Auto-register when module is imported
register_s3_handler()