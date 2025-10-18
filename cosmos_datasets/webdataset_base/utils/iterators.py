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

import io
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from typing import IO, Any, BinaryIO
from urllib.parse import urlparse

import pandas as pd
import webdataset.gopen as gopen_webdata
import yaml
from webdataset import cache, filters, shardlists
from webdataset.compat import FluidInterface
from webdataset.handlers import reraise_exception
from webdataset.pipeline import DataPipeline
from webdataset.pytorch import IterableDataset
from webdataset.tariterators import group_by_keys, tar_file_iterator

from cosmos_datasets.webdataset_base.config.schema import TarSample

import logging

logger = logging.getLogger(__name__)

# S3 streaming support
def _gopen_s3(url: str, mode: str = "rb", bufsize: int = 8192, **kwargs):
    """Handle S3 URLs for streaming"""
    if mode != "rb":
        raise ValueError(f"S3 streaming only supports 'rb' mode, got: {mode}")
    
    try:
        # Import S3 support
        from cosmos_datasets.s3_support import gopen_s3
        return gopen_s3(url, mode, bufsize, **kwargs)
    except ImportError:
        raise ImportError("boto3 is required for S3 support. Install with: pip install boto3")
    except Exception as e:
        logger.error(f"Failed to open S3 stream {url}: {e}")
        raise


def gopen(url: tuple | str, mode: str = "rb", bufsize: int = 8192, **kw) -> io.BytesIO | BinaryIO | IO:
    r"""Open the URL.
    This uses the `gopen_schemes` dispatch table to dispatch based
    on scheme.
    Support for the following schemes is built-in: pipe, file,
    http, https, sftp, ftps, scp.
    When no scheme is given the url is treated as a file.
    You can use the OPEN_VERBOSE argument to get info about
    files being opened.
    Args:
        url (list[str]): the source URL
        mode (str): the mode ("rb", "r")
        bufsize (int): the buffer size
    Returns:
        Byte streams
    """
    global fallback_gopen
    verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
    if verbose:
                logger.info(f"GOPEN {url} {gopen_webdata.info}")

    assert mode in ["rb", "wb"], mode
    if url == "-":
        if mode == "rb":
            return sys.stdin.buffer
        elif mode == "wb":
            return sys.stdout.buffer
        else:
            raise ValueError(f"unknown mode {mode}")

    # Handle tuple input (url_path, dset_id) - extract just the path
    if isinstance(url, tuple):
        url = url[0]  # Use only the path, ignore dset_id
    
    # For all other gopen schemes, use the native webdataset gopen functions.
    # pr = gopen_webdata.urlparse(url)
    assert isinstance(url, str)
    pr = urlparse(url)
    if pr.scheme == "":
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(url, mode, buffering=bufsize)
    if pr.scheme == "file":
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(pr.path, mode, buffering=bufsize)
    
    # Handle S3 URLs natively
    if pr.scheme == "s3":
        return _gopen_s3(url, mode, bufsize, **kw)
    
    handler = gopen_webdata.gopen_schemes["__default__"]
    handler = gopen_webdata.gopen_schemes.get(pr.scheme, handler)
    return handler(url, mode, bufsize, **kw)  # type: ignore


def url_opener(data: Iterable, handler: Callable = reraise_exception, **kw) -> Iterator[dict]:
    r"""Given a stream of url names (packaged in `dict(url=url)`), yield opened streams.

    Args:
        data (Iterable): Iterator of dictionaires containing url paths.
        handler (Callable): Exception handler.

    Yields:
      Dictionaries with this structure:
        {"url": ...
         "stream": list[Union[io.BytesIO, RetryingStream]]}
    """
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        assert isinstance(url, TarSample), "URL should be of type TarSample"
        try:
            stream = []
            for data_key in url.keys:
                # For multi-modal datasets, construct path as root/data_key/path
                url_path_full = os.path.join(url.root, data_key, url.path)
                url_key = (url_path_full, url.dset_id)  # Pass tuple like original
                stream.append(gopen(url_key, **kw))

            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            logger.info(f"Got an exception while opening urls - {exn}")
            exn.args = exn.args + (url,)  # noqa: RUF005
            if handler(exn):
                continue
            else:
                break


def process_sample(sample, url, key_idx):
    assert isinstance(sample, dict) and "data" in sample and "fname" in sample
    # Edit the url entries
    sample["__url__"] = url
    # This is the folder name
    data_key = url.keys[key_idx]
    # Handle the case where data_key has "/"
    data_key = data_key.replace("/", "_")
    # Edit the fname to include the data_key
    prefix, suffix = sample["fname"].split(".")  # {sample_key}.{suffix} e.g. "id_1410095.json"

    # e.g. "id_1410095.caption_ai_from_image.json"
    sample["fname"] = f"{prefix}.{data_key}.{suffix}"

    return sample


def _synchronized_stream_processor(tar_file_iterator_list, url, handler):
    """
    Process multiple tar file iterators in a synchronized manner to prevent
    sample order mismatch when one stream fails and continues.
    
    This function ensures that when one stream encounters an error and continues
    with the next sample, all other streams are also advanced to maintain
    sample alignment across streams.
    
    Args:
        tar_file_iterator_list: List of tar_file_iterator objects
        url: URL object containing stream information
        handler: Exception handler function
        
    Yields:
        Processed samples in synchronized order
    """
    # Create synchronized iterators that handle errors consistently
    synchronized_iterators = []
    
    for stream_id, iterator in enumerate(tar_file_iterator_list):
        def make_sync_iterator(iter_obj, stream_idx):
            """Create a synchronized iterator that handles errors consistently"""
            try:
                for item in iter_obj:
                    yield item
            except Exception as exn:
                logger.warning(f"Stream {stream_idx} encountered error: {exn}")
                # When one stream fails, we need to ensure all streams advance together
                # This prevents sample order mismatch
                # Continue with next sample, but mark this stream as having an error
                # so other streams can catch up
                yield None
                #if handler(exn):
                #    yield None  # Placeholder to maintain synchronization
                #else:
                #    # If handler says to stop, re-raise the exception
                #    raise exn
        
        synchronized_iterators.append(make_sync_iterator(iterator, stream_id))
    
    # Process samples in synchronized batches
    while True:
        try:
            # Get next sample from all streams
            sample_batch = []
            all_streams_exhausted = True
            
            for sync_iter in synchronized_iterators:
                try:
                    sample = next(sync_iter)
                    sample_batch.append(sample)
                    all_streams_exhausted = False
                except StopIteration:
                    # This stream is exhausted
                    all_streams_exhausted = True
                    break
                except Exception as exn:
                    logger.warning(f"Error in synchronized processing: {exn}")
                    sample_batch.append(None)  # Placeholder for failed stream
                    all_streams_exhausted = False
            
            # If all streams are exhausted, we're done
            if all_streams_exhausted:
                break
                
            # Process the synchronized batch
            # Skip if ANY sample is None (any stream failed) - only yield when ALL streams succeed
            if any(sample is None for sample in sample_batch):
                logger.warning(f"Skipping batch due to failed streams")
                continue
                
            # Yield samples from the synchronized batch
            # Only yield when ALL streams succeeded (no None values)
            for key_idx, sample_key in enumerate(sample_batch):
                sample_key = process_sample(sample_key, url, key_idx)
                yield sample_key
                    
        except StopIteration:
            break
        except Exception as exn:
            logger.error(f"Error in synchronized stream processing: {exn}")
            if handler(exn):
                continue
            else:
                break


def _legacy_stream_processor(tar_file_iterator_list, url, handler):
    """
    Legacy stream processing that uses zip() - may cause sample order mismatch
    when one stream fails and continues while others don't.
    
    This is kept for backward compatibility and when synchronized processing
    is not needed.
    """
    if url.sample_keys_full_list is None:  # Original behavior
        # tar_file_iterator_list is a list of iterator: [tar_file_iterator_0, tar_file_iterator_1, ... tar_file_iterator_N]
        for sample in zip(*tar_file_iterator_list, strict=False):
            # Merging data from all streams
            # sample is list of dictionaries, each dictionary contains data and fname
            # sample [tar_file_iterator_0[0], tar_file_iterator_1[0], ... tar_file_iterator_N[0]], length = num_of_data_key
            for key_idx, sample_key in enumerate(sample):
                sample_key = process_sample(sample_key, url, key_idx)
                yield sample_key
    else:
        # Provide fallback to standard processing
        for sample in zip(*tar_file_iterator_list, strict=False):
            for key_idx, sample_key in enumerate(sample):
                sample_key = process_sample(sample_key, url, key_idx)
                yield sample_key


def tar_file_expander(
    data: Iterable[dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Callable[[str], bool] | None = None,
    rename_files: Callable[[str], str] | None = None,
    synchronized_processing: bool = True,
) -> Iterator[dict[str, Any]]:
    """Expand tar files.

    Args:
        data (Iterable[Iterable[Dict[str, Any]]]): iterator over opened tar file streams.
        handler (Callable[[Exception], bool]): exception handler.
        select_files (Optional[Callable[[str], bool]]): select files from tarfiles by name (permits skipping files).
        rename_files (Optional[Callable[[str], bool]]): Renaming tar files.
        synchronized_processing (bool): If True, use synchronized processing to prevent sample order mismatch
                                       when one stream fails and continues. If False, use legacy zip() processing.

    Yields:
        a stream of samples.
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            tar_file_iterator_list = []
            for stream_id in range(len(source["stream"])):
                tar_file_iterator_list.append(
                    tar_file_iterator(
                        source["stream"][stream_id],
                        handler=handler,
                        select_files=select_files,
                        rename_files=rename_files,
                    )
                )
            
            # Choose processing method based on configuration
            if synchronized_processing:
                # Use synchronized processing to prevent sample order mismatch
                for sample_key in _synchronized_stream_processor(tar_file_iterator_list, url, handler):
                    yield sample_key
            else:
                # Use legacy processing (may cause sample order mismatch)
                for sample_key in _legacy_stream_processor(tar_file_iterator_list, url, handler):
                    yield sample_key

        except Exception as exn:
            logger.info(f"Got an exception while expanding tars - {exn}")
            exn.args = exn.args + (source.get("stream"), source.get("url"))  # noqa: RUF005
            if handler(exn):
                continue
            else:
                break


def correct_order(sample_list: list[dict], expected_keys_order: list[str]) -> list[dict]:
    """Make sure the order of samples are the same as the url.keys order."""
    data_keys_per_sample = [sample["fname"].split(".")[1] for sample in sample_list]
    expected_keys_order = [key.replace("/", "_") for key in expected_keys_order]
    if data_keys_per_sample == expected_keys_order:  # Correct order
        return sample_list
    # Order the sample_list based on the expected_keys_order
    sample_list_ordered = [None] * len(expected_keys_order)
    for data_key, sample in zip(data_keys_per_sample, sample_list, strict=False):
        idx = expected_keys_order.index(data_key)
        sample_list_ordered[idx] = sample
    return sample_list_ordered


def load_func_parquet(buffer):
    data_list = pd.read_parquet(buffer).values.tolist()
    names = [data[0] for data in data_list]
    return names


def tarfile_samples(
    src: Iterable,
    handler: Callable = reraise_exception,
    streaming_download: bool = True,
    synchronized_processing: bool = True,
) -> Iterator[dict]:
    r"""
    Given an iterator of filenames, this function opens the URL streams
    and groups data by keys.

    Args:
        src (Iterable): Iterator of TarSample.
        handler (Callable): Exception handler.
        streaming_download(bool): If enabled, performs streaming download.
        synchronized_processing (bool): If True, use synchronized processing to prevent sample order mismatch
                                       when one stream fails and continues. If False, use legacy zip() processing.
    """
    streams = url_opener(
        src,
        handler=handler,
        streaming_download=streaming_download,
    )
    files = tar_file_expander(streams, handler=handler, synchronized_processing=synchronized_processing)
    samples = group_by_keys(files, handler=handler)
    
    # Custom sample processing - you can manipulate samples here
    def process_samples(sample_iterator):
        for sample in sample_iterator:
            yield sample
    
    return process_samples(samples)


tarfile_to_samples = filters.pipelinefilter(tarfile_samples)


class WebDataset(DataPipeline, FluidInterface):
    r"""Webdataset class."""

    def __init__(
        self,
        urls: list[TarSample],
        handler: Callable = reraise_exception,
        resampled: bool = False,
        shardshuffle: bool | None = None,
        cache_size: int = -1,
        cache_dir: str | None = None,
        detshuffle: bool = False,
        nodesplitter: Callable = shardlists.single_node_only,
        verbose: bool = False,
        streaming_download: bool = True,
        synchronized_processing: bool = True,
    ):
        r"""
        Args:
            urls (list[TarSample]): An iterator containing a list of url names.
            handler (Callable): Exception handler.
            resampled (bool): If true, sample shards from shard list with replacement.
            shardshuffle (bool): If true, shuffles the entire shard list.
            cache_size (int): Size of cache.
            cache_dir (str): Path to store cache.
            detshuffle (bool): Whether to use deterministic shuffling when shardshuffle is True.
            nodesplitter (Callable): Function for splitting urls among nodes.
            verbose (bool): If True, prints logs.
            streaming_download (bool): Whether to do streaming download or full object download.
            synchronized_processing (bool): If True, use synchronized processing to prevent sample order mismatch
                                           when one stream fails and continues. If False, use legacy zip() processing.
        """
        super().__init__()
        if isinstance(urls, IterableDataset):
            assert not resampled
            self.append(urls)
        elif isinstance(urls, str) and (urls.endswith(".yaml") or urls.endswith(".yml")):
            with open(urls) as stream:
                spec = yaml.safe_load(stream)
            assert "datasets" in spec
            self.append(shardlists.MultiShardSample(spec))
        elif isinstance(urls, dict):
            assert "datasets" in urls
            self.append(shardlists.MultiShardSample(urls))
        elif resampled:
            self.append(shardlists.ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is True:
                shardshuffle = 100  # type: ignore
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))
        if cache_dir is None or cache_size == 0:
            self.append(
                tarfile_to_samples(
                    handler=handler,
                    streaming_download=streaming_download,
                    synchronized_processing=synchronized_processing,
                )
            )
        else:
            # We dont use cache.
            assert cache_size == -1 or cache_size > 0
            self.append(
                cache.cached_tarfile_to_samples(
                    handler=handler,
                    verbose=verbose,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                )
            )
