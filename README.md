# Cosmos-Dataloader
Cosmos-predict2 has a nicely designed dataloader and customized webdatset. I port the dataloader related code into this standalone repo with minimal requirements. So we can use it for other projects without the need to deal with all those dependencies from diffusion and megatron

## Table of Contents
1. [Quick Start](#quick-start)
2. [Multiple Tar Loading Implementation](#multiple-tar-loading-implementation)
3. [Video Processing Architecture](#video-processing-architecture)
4. [Augmentation System](#augmentation-system)
5. [Distribution Strategies](#distribution-strategies)
6. [Distributed Training Implementation](#distributed-training-implementation)
7. [CachedReplayDataLoader](#cachedreplaydataloader)
8. [S3 Integration](#s3-integration)
9. [Performance Optimizations](#performance-optimizations)

## Quick Start

### Overview
There are three main examples in the `examples/` folder that demonstrate different use cases and capabilities getting started with this dataloader.

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd cosmos-dataloader

# Install in development mode
pip install -e .

# Or install with dependencies
pip install -e .[dev]
```

### Basic Usage
After installation, you can import and use the dataloader:

```python
import cosmos_datasets
from cosmos_datasets.webdataset import Dataset
from cosmos_datasets.webdataset_base.dataloader import DataLoader
from cosmos_datasets.webdataset_base.config.schema import DatasetConfig
from cosmos_datasets.dataset_utils import create_image_dataset_with_real_images

# Create a dataset
dataset = Dataset(...)
dataloader = DataLoader(dataset, ...)
```

### Example 1: Image Dataset Testing (`test_image_cosmos_webdataset.py`)

#### Purpose
Demonstrates how to work with image datasets using cosmos-datasets format, including basic image loading, augmentation, and multi-aspect ratio handling. The example first creates a dummy dataset of empty images that follows the format required by the dataloader, then demonstrate dataloading examples. Optional: S3 or local

#### Usage
```bash
# Run basic image dataset test
python examples/test_image_cosmos_webdataset.py

# Test with S3 (optional)
python examples/test_image_cosmos_webdataset.py --s3-bucket my-bucket --s3-profile my-profile
```

### Example 2: Video Dataset Testing (`test_video_cosmos_webdataset.py`)

#### Purpose
Similar to image example, this shows how to work with video datasets, including video decoding, chunked processing, and video-specific augmentations. 

#### Usage
```bash
# Run basic video dataset test
python examples/test_video_cosmos_webdataset.py

# Test multi-aspect ratio video dataset
python examples/test_video_cosmos_webdataset.py --multi-aspect-ratio

# Test with S3 (optional)
python examples/test_video_cosmos_webdataset.py --s3-bucket my-bucket
```

### Example 3: WebDataset Conversion (`convert_webdataset.py`)

#### Purpose
Provides a conversion tool to transform conventional WebDataset format into cosmos-datasets format. Optional s3 or local

#### Usage
```bash
# Convert local dataset
python examples/convert_webdataset.py \
    --input-path /path/to/conventional/dataset \
    --output-dir /path/to/cosmos/dataset \
    --extension-mapping '{"txt":"captions","json":"metas"}' \
    --chunk-size 100

# Convert S3 dataset
python examples/convert_webdataset.py \
    --input-path s3://my-bucket/conventional-dataset \
    --output-dir s3://my-bucket/cosmos-dataset \
    --s3-profile my-profile \
    --extension-mapping '{"txt":"captions","json":"metas"}'
```

## Multiple Tar Loading Implementation

### Overview
Cosmos-datasets implements a multiple tar loading system that separates modalities into different tar files while maintaining synchronization across samples. This allows for modular data organization and efficient updates. When adding a new modality or making changes to existing ones, we only need to add a new directory of tar files without the need to re-build the entire datset. 

Conventional WebDataset Format:
```
dataset/
├── 000000.tar  # Contains: 000000.jpg, 000000.txt, 000000.json
├── 000001.tar  # Contains: 000001.jpg, 000001.txt, 000001.json
└── ...
```

Cosmos Datasets Format:
```
dataset/
├── images/
│   └── part_000/
│       └── 000.tar  # Contains: 000000.jpg, 000001.jpg, ...
├── captions/
│   └── part_000/
│       └── 000.tar  # Contains: 000000.txt, 000001.txt, ...
├── metas/
│   └── part_000/
│       └── 000.tar  # Contains: 000000.json, 000001.json, ...
└── wdinfo.json  # Dataset metadata
```

### Core Components

#### 1. TarSample Structure
a `TarSample` contain one or multiple modalities. Each modality is a subdirector of tar files like `images`, `captions`, `metas` in this example:

```
dataset/
├── images/
│   └── part_000/
│       └── 000.tar  # Contains: 000000.jpg, 000001.jpg, ...
├── captions/
│   └── part_000/
│       └── 000.tar  # Contains: 000000.txt, 000001.txt, ...
├── metas/
│   └── part_000/
│       └── 000.tar  # Contains: 000000.json, 000001.json, ...
└── wdinfo.json  # Dataset metadata
```


```python
@dataclass
class TarSample:
    path: str                    # Path to the tar file (e.g., "part_000/000.tar")
    root: str                    # Root directory (e.g., "/data/dataset/")
    keys: List[str]              # Modality to load from this tar file(e.g., ["images", "captions", "metas"])
    meta: DatasetInfo            # Dataset metadata
    dset_id: str                 # Dataset identifier
    sample_keys_full_list: Optional[str] = None
```

#### 2. URL Opening Process (`url_opener`)
The system opens multiple streams simultaneously for each subdirectory of tar files:

```python
def url_opener(data: Iterable, handler: Callable = reraise_exception, **kw) -> Iterator[dict]:
    for sample in data: # loop through all TarSample
        url = sample["url"]  # TarSample object
        stream = []
        for data_key in url.keys:  # For each modality inside this subdirectory
            # Construct full path: root/data_key/path
            url_path_full = os.path.join(url.root, data_key, url.path)
            url_key = (url_path_full, url.dset_id)
            stream.append(gopen(url_key, **kw))  # Open stream for this modality
        
        sample.update(stream=stream)  # Store all streams
        yield sample
```

**Key insight**: For a single sample, this creates multiple streams:
- `stream[0]` → `root/images/part_000/000.tar`
- `stream[1]` → `root/captions/part_000/000.tar` 
- `stream[2]` → `root/metas/part_000/000.tar`

#### 3. Tar File Expansion (`tar_file_expander`)
This is the core of multiple tar loading. It processes multiple tar files in parallel:

```python
def tar_file_expander(data: Iterable[dict[str, Any]], ...) -> Iterator[dict[str, Any]]:
    for source in data:
        url = source["url"]
        tar_file_iterator_list = []
        
        # Create iterator (which handles the url_open as above) for each modality tar file
        for stream_id in range(len(source["stream"])):
            tar_file_iterator_list.append(
                tar_file_iterator(
                    source["stream"][stream_id],  # Each stream is a different modality
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
                )
            )
        
        # CRITICAL: Use zip to synchronize across modalities
        for sample in zip(*tar_file_iterator_list, strict=False):
            # sample is a list: [images_data, captions_data, metas_data]
            for key_idx, sample_key in enumerate(sample):
                sample_key = process_sample(sample_key, url, key_idx)
                yield sample_key
```

#### 4. Synchronization Mechanism
The key to multiple tar loading is the `zip(*tar_file_iterator_list, strict=False)` operation:

- **`tar_file_iterator_list[0]`**: Iterator over `images/part_000/000.tar` → yields `000000.jpg`, `000001.jpg`, ...
- **`tar_file_iterator_list[1]`**: Iterator over `captions/part_000/000.tar` → yields `000000.txt`, `000001.txt`, ...
- **`tar_file_iterator_list[2]`**: Iterator over `metas/part_000/000.tar` → yields `000000.json`, `000001.json`, ...

The `zip()` function ensures that:
- `sample[0]` contains data from `000000.jpg`
- `sample[1]` contains data from `000000.txt` 
- `sample[2]` contains data from `000000.json`

This creates a **synchronized sample** where all modalities correspond to the same sample ID.

#### 5. Sample Processing (`process_sample`)
Each sample gets processed to add modality information:

```python
def process_sample(sample, url, key_idx):
    sample["__url__"] = url
    # Add modality key (images, captions, metas)
    data_key = url.keys[key_idx]
    sample["fname"] = f"{prefix}.{data_key}.{suffix}"
    return sample
```

#### 6. Grouping by Keys (`group_by_keys`)
Finally, the WebDataset's `group_by_keys` function groups all the individual modality samples back into a batch of samples:

```python
# Before group_by_keys:
# Sample 1: {"data": image_bytes, "fname": "000000.images.jpg"}
# Sample 2: {"data": caption_text, "fname": "000000.captions.txt"}  
# Sample 3: {"data": metadata_json, "fname": "000000.metas.json"}

# After group_by_keys:
# Single sample: {
#   "images": image_bytes,
#   "captions": caption_text, 
#   "metas": metadata_json
# }
```

### Benefits of This Approach

- **Modularity**: Each modality is in its own tar file
- **Efficiency**: Only loads needed modalities
- **Synchronization**: Ensures all modalities correspond to same sample ID i.e. base name
- **Scalability**: Can handle any number of modalities
- **Streaming**: Supports streaming downloads for large datasets

## Video Processing Architecture

### Chunked Video Processing
Videos are processed in chunks with corresponding captions:

```python
# Video metadata structure
{
  "t2w_windows": [
    {
      "caption": "A person walking in the park",
      "start_frame": 0,
      "end_frame": 120
    },
    {
      "caption": "The person sits on a bench", 
      "start_frame": 120,
      "end_frame": 240
    }
  ],
  "fps": 30,
  "nb_frames": 300
}
```

### Video Decoders

#### 1. `video_naive_bytes`
Returns raw video bytes - to be paired with the `video_parsing.py` to handle chunk-wise captions with various chunk lengths.

#### 2. `chunked_video_decoder`
Processes chunked videos with captions of fixed chunk length.

#### 3. `chunked_video_decoder_with_fixed_fps`
Fixed FPS processing with optimized frame sampling.

#### 4. `chunked_video_decoder_w_lower_fps`
Lower FPS optimization for efficient processing.

### Video Parsing Augmentor
The `VideoParsing` augmentor handles:
- Chunk selection from video
- Frame sampling within chunks
- Caption matching to video segments
- Temporal subsampling
- To be paired with `video_naive_bytes` decoder

## Augmentation System
The augmentation system loop through all augumentors and apply them one by one. An augumentor does some transforms on one or multiple modaliaties in the sample, e.g. apply padding or cropping on images, extracing edge maps etc

```python
@staticmethod
def augmentor_fn(data, augmentations):
    # Build augmentor chain
    for aug_fn in augmentations:
        # Use generator function as augmentor
        if getattr(aug_fn, "is_generator", False):
            data = aug_fn(data)
        else:  # Use regular function as augmentor (backward compatibility)
            data = wrap_augmentor_func_as_generator(aug_fn, data)
    yield from data
```

### Built-in Augmentors

#### Image Augmentors
- **`ResizeLargestSideAspectPreserving`**: Maintains aspect ratio while resizing
- **`ReflectionPadding`**: Adds reflection padding to images
- **`Normalize`**: Standard normalization (mean=0.5, std=0.5)
- **`AppendFPSFramesForImage`**: Adds FPS information to image metadata to make image batch compatible with vidoe mixed training

#### Video Augmentors
- **`VideoParsing`**: Chunk-based video processing with caption matching
- **`UniformTemporalSubsample`**: Temporal sampling for video frames

## Distribution Strategies

### Overview
The distributors in cosmos-datasets are designed to handle distributed training scenarios efficiently, particularly for multi-GPU and multi-node setups. They implement sophisticated strategies to ensure proper data distribution, worker synchronization, and aspect ratio consistency across distributed workers.

### Goals of Distributed Training

#### 1. **Worker Isolation by Aspect Ratio**
This `ShardlistMultiAspectRatio` distributor handles the multi-aspect ratio case. For the dataloader to be successful,each worker should load only one aspect ratio. Else, there can be a batch where two aspect ratios would be present which would raise an error in collate function. So, we design data distribution strategy so that each worker sees only one aspect ratio.


#### 2. **DDP (Distributed Data Parallel) Equalization**
Ensures that all workers receive the same number of tar files, preventing some workers from finishing early while others are still processing data. This is crucial for proper epoch termination in distributed training.

#### 3. **Efficient Resource Utilization**
Maximizes GPU utilization by distributing work evenly across all available workers while maintaining data consistency.

### Distribution Strategies

#### 1. **Basic Distributor (`ShardlistBasic`)**

**Simple Round-Robin Distribution**
```python
# Split by node
if self.split_by_node:
    urls = urls[rank::world_size]

# Split by worker
if self.split_by_worker:
    urls = urls[worker_id::num_workers]
```

**URL Extension for Equalization**
```python
if self.repeat_url:
    nworkers_all = world_size * num_workers
    num_urls_per_process = (num_urls + nworkers_all - 1) // nworkers_all
    extended_url_list_size = num_urls_per_process * nworkers_all
    urls = repeat_list(urls, extended_url_list_size)
```

**Goal**: Ensures each worker receives the same number of batches by extending the URL list.

#### 2. **Multi-Aspect Ratio Distributor (`ShardlistMultiAspectRatio`)**

**Aspect Ratio Splitting**
```python
def _split_urls_by_aspect_ratio(self):
    url_aspect_split = defaultdict(list)
    
    for url in self.urls:
        dset_info = url.meta
        aspect_ratio = dset_info.opts["aspect_ratio"]
        url_aspect_split[aspect_ratio].append(url)
```

**Goal**: Groups tar files by aspect ratio to ensure workers only process one aspect ratio.

**DDP Equalization Algorithm**
```python
def _ddp_equalize(self, url_aspect_split, nworkers_all):
    betas = []
    n_total = sum([len(url_aspect_split[aspect_ratio]) for aspect_ratio in url_aspect_split])
    
    # Initial assignment based on proportion
    for i, aspect_ratio in enumerate(url_aspect_split):
        betas.append(math.ceil((len(url_aspect_split[aspect_ratio]) / n_total) * nworkers_all))
    
    # Constraint: total workers must equal nworkers_all
    betas[aspect_ind_with_most_elems] += nworkers_all - sum(betas)
    
    # Rebalance URLs
    num_urls_per_worker = math.ceil(n_total / sum(betas))
    for i, aspect_ratio in enumerate(url_aspect_split):
        url_aspect_split[aspect_ratio] = repeat_list(
            url_aspect_split[aspect_ratio], 
            betas[i] * num_urls_per_worker
        )
```

**Goal**: Ensures each worker gets the same number of tar files by repeating URLs proportionally.

**Worker-URL Mapping**
```python
def _obtain_node_worker_url_mapping(self, url_aspect_split, num_urls_per_worker, 
                                  rank, world_size, worker_id, num_workers):
    # First chunk the tars by aspect ratio
    chunk_mappings = []
    for aspect_ratio in url_aspect_split:
        samples_asp = url_aspect_split[aspect_ratio]
        nchunks_asp = int(len(samples_asp) / num_urls_per_worker)
        for chunk_id in range(nchunks_asp):
            chunk_mappings.append((aspect_ratio, samples_asp[chunk_id::nchunks_asp]))
    
    # Split by rank and workers
    chunk_mappings = chunk_mappings[rank::world_size]
    chunk_mappings = chunk_mappings[worker_id::num_workers]
    
    # Each worker gets exactly one aspect ratio
    assert len(chunk_mappings) == 1
    return chunk_mappings[0][1]
```

**Goal**: Ensures each worker processes exactly one aspect ratio.

#### 3. **Multi-Aspect Ratio Infinite (`ShardlistMultiAspectRatioInfinite`)**

**Worker Allocation to Aspect Ratios**
```python
def _allocate_workers_to_aspects(self, url_aspect_split, nworkers_all):
    aspect_worker_allocation = []
    
    for aspect_ratio in url_aspect_split:
        num_samples = len(url_aspect_split[aspect_ratio])
        num_workers_for_aspect = max(1, int((num_samples / total_samples) * nworkers_all))
        aspect_worker_allocation.append((aspect_ratio, num_workers_for_aspect))
    
    return aspect_worker_allocation
```

**Goal**: Dynamically allocates workers to aspect ratios based on data volume.

### Key Features

#### 1. **Resume Functionality**
```python
if self.resume_flag:
    self.epoch = int(os.environ.get("WDS_EPOCH_NUM", 0))
    self.start_index = int(os.environ.get("WDS_START_INDEX", 0)) // self.chunk_size
```

**Goal**: Allows training to resume from a specific epoch and iteration from arround the same place in the webdatset.

#### 2. **Deterministic Shuffling**
```python
if self.shuffle:
    random.Random(rank * world_size + worker_id * num_workers).shuffle(urls)
```

**Goal**: Ensures reproducible shuffling across workers while maintaining different random seeds.

#### 3. **Infinite Loading Support**
```python
if self.is_infinite_loader:
    while True:
        cur_time = time.time_ns()
        random.Random(cur_time).shuffle(url_list)
        for url in url_list:
            yield dict(url=url)
```

**Goal**: Supports continuous training scenarios without epoch boundaries.

### Error Handling

#### 1. **Aspect Ratio Validation**
```python
if "aspect_ratio" not in dset_info.opts:
    raise ValueError("aspect_ratio should be specified in dataset_info when using multi aspect distributor")
```

#### 2. **Worker Count Validation**
```python
assert len(chunk_mappings) == 1  # Each worker gets exactly one aspect ratio
```

#### 3. **DDP Termination**
- Ensures all workers finish at the same time
- Prevents hanging in distributed training
- Maintains training consistency

## CachedReplayDataLoader

### Overview
The `CachedReplayDataLoader` is a sophisticated wrapper around PyTorch DataLoaders that implements asynchronous caching and replay of augmented data batches. It's designed to mitigate slow data loading issues by prefetching and caching multiple augmented versions of each batch in the background.

### Key Features

#### 1. **Asynchronous Background Prefetching**
- Runs a separate daemon thread that continuously fetches batches from the underlying DataLoader
- Applies augmentation functions to generate multiple augmented versions
- Stores augmented batches in a thread-safe cache

#### 2. **Smart Batch Concatenation**
- Supports concatenating multiple batches into larger batches
- Handles different data types (tensors, strings, lists) intelligently
- Maintains batch structure while increasing effective batch size

#### 3. **Memory Management**
- Controlled cache size to prevent memory overflow
- Thread-safe cache access with condition variables
- Automatic cleanup and error handling

#### 4. **Performance Monitoring**
- Built-in `OperationWatchdog` for performance tracking
- Monitors fetch times, augmentation times, and cache operations
- Identifies bottlenecks in the data loading pipeline

### Architecture

#### Core Components

**1. Main Thread (Data Consumer)**
```python
def __iter__(self) -> Iterator[dict]:
    """Yield augmented data batches from the cache"""
    while not self._stop_event.is_set():
        # Wait for cache to have data
        with self._cache_cond:
            while not self._cache and not self._stop_event.is_set():
                self._cache_cond.wait(timeout=1.0)
        
        # Get batch from cache
        idx = self.rng.integers(0, len(self._cache))
        batch = self._cache.pop(idx)
        yield batch
```

**2. Background Thread (Data Producer)**
```python
def _prefetch_loop(self) -> None:
    """Continuously fetch and augment batches"""
    while not self._stop_event.is_set():
        # Fetch raw batch
        batch = next(self._data_iter)
        
        # Apply augmentation
        augmented_batches = self.cache_augmentation_fn(batch)
        
        # Store in cache
        for aug_batch in augmented_batches:
            with self._cache_cond:
                while len(self._cache) >= self.cache_size:
                    self._cache_cond.wait()
                self._cache.append(aug_batch)
```

**3. Thread-Safe Cache Management**
```python
# Cache access is protected by condition variables
self._cache_cond = threading.Condition()
self._cache: list[dict] = []
```

### Smart Batch Concatenation

#### Concatenation Algorithm
```python
def concatenate_batches(n: int, data_batches: list[dict]) -> list[dict]:
    """Smartly concatenate n input data batches into m output data batches"""
    
    # Process in groups of n
    for i in range(m):
        batches_to_concat = []
        for j in range(n):
            batch_idx = j * m + i
            batches_to_concat.append(data_batches[batch_idx])
        
        # Merge dictionaries
        merged_batch = {}
        for key in all_keys:
            values = [batch[key] for batch in batches_to_concat if key in batch]
            
            if isinstance(first_value, torch.Tensor):
                merged_batch[key] = torch.cat(values, dim=0)
            elif isinstance(first_value, str):
                merged_batch[key] = values[0]  # Take first string
            elif isinstance(first_value, list):
                merged_list = []
                for v in values:
                    merged_list.extend(v)
                merged_batch[key] = merged_list
```

#### Data Type Handling
- **Tensors**: Concatenated along dimension 0 using `torch.cat()`
- **Strings**: First string is used (no concatenation)
- **Lists**: Extended with all values from all batches
- **Other types**: Stored as list of values

### Performance Optimizations

#### 1. **Asynchronous Processing**
- Background thread handles slow I/O operations
- Main thread never blocks on data loading
- Continuous prefetching keeps cache populated

#### 2. **Memory Management**
- Controlled cache size prevents memory overflow
- Thread-safe access with condition variables
- Automatic cleanup on close

#### 3. **Error Handling**
- Comprehensive exception handling in background thread
- Error propagation to main thread
- Graceful shutdown with timeout

#### 4. **Performance Monitoring**
```python
self._watchdog = OperationWatchdog(
    warning_threshold=100,  # Warn if operation takes >100ms
    verbose_interval=600,   # Print stats every 10 minutes
    name=name
)

# Monitor different operations
with self._watchdog.watch("fetch raw batch", verbose_first_n=5):
    batch = next(self._data_iter)

with self._watchdog.watch("augmentation", verbose_first_n=5):
    augmented_batches = self.cache_augmentation_fn(batch)
```

### Usage Examples

#### 1. **Basic Usage**
```python
# Create augmentation function
def cache_augment_fn(batch):
    # Return multiple augmented versions
    return [augment(batch) for _ in range(4)]

# Create cached replay dataloader
cached_dataloader = CachedReplayDataLoader(
    data_loader=base_dataloader,
    cache_size=32,
    cache_augmentation_fn=cache_augment_fn,
    concat_size=1
)

# Use in training loop
for batch in cached_dataloader:
    # Process batch
    pass
```

#### 2. **With Batch Concatenation**
```python
# Concatenate 4 batches into 1 larger batch
cached_dataloader = CachedReplayDataLoader(
    data_loader=base_dataloader,
    cache_size=64,
    cache_augmentation_fn=cache_augment_fn,
    concat_size=4  # Concatenate 4 batches
)
```

#### 3. **Factory Function**
```python
# Use the factory function
dataloader = get_cached_replay_dataloader(
    use_cache=True,
    cache_size=32,
    concat_size=2,
    cache_augment_fn=my_augment_fn,
    batch_size=16,
    num_workers=4
)
```

### Configuration Options

#### 1. **Cache Parameters**
- `cache_size`: Maximum number of augmented batches to store
- `concat_size`: Number of batches to concatenate
- `cache_augmentation_fn`: Function to create augmented versions

#### 2. **Performance Tuning**
- `warning_threshold`: Operation time threshold for warnings
- `verbose_interval`: Interval for performance statistics
- `timeout`: Timeout for thread operations

#### 3. **Error Handling**
- Automatic exception propagation
- Graceful shutdown with timeout
- Comprehensive error context

### Benefits

#### 1. **Performance Improvements**
- Hides I/O latency through prefetching
- Reduces training time by eliminating data loading bottlenecks
- Enables higher GPU utilization

#### 2. **Memory Efficiency**
- Controlled cache size prevents memory overflow
- Smart batch concatenation reduces memory fragmentation
- Automatic cleanup prevents memory leaks

#### 3. **Flexibility**
- Works with any DataLoader
- Configurable augmentation functions
- Support for different batch sizes and concatenation strategies

#### 4. **Reliability**
- Comprehensive error handling
- Thread-safe operations
- Graceful shutdown and cleanup

### Best Practices

#### 1. **Cache Size Tuning**
- Start with cache_size = 32-64
- Monitor memory usage and adjust accordingly
- Larger cache for slower augmentation functions

#### 2. **Augmentation Function Design**
- Return multiple diverse augmented versions
- Keep augmentation functions fast
- Handle errors gracefully

#### 3. **Batch Concatenation**
- Use concat_size > 1 for larger effective batch sizes
- Ensure batch_size is divisible by concat_size
- Monitor memory usage with larger concatenated batches

#### 4. **Performance Monitoring**
- Use OperationWatchdog to identify bottlenecks
- Monitor cache hit rates and prefetch times
- Adjust parameters based on performance metrics

### Troubleshooting

#### 1. **Memory Issues**
- Reduce cache_size
- Decrease concat_size
- Monitor memory usage with profiling tools

#### 2. **Performance Issues**
- Check augmentation function performance
- Monitor I/O operations
- Adjust cache_size based on throughput

#### 3. **Threading Issues**
- Ensure proper cleanup with close()
- Check for deadlocks in augmentation functions
- Monitor thread status and errors

## S3 Integration

### S3 Configuration
```python
def configure_s3_defaults(
    profile_name: str = "default",
    endpoint_url: str = None,
    region_name: str = "us-east-1"
):
    """Configure S3 defaults for the dataset"""
    # Implementation details
```
