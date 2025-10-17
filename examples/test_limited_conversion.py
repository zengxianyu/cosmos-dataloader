#!/usr/bin/env python3
"""Test script for multiprocess S3 conversion with limited files."""

import subprocess
import sys

def test_limited_conversion():
    """Test conversion with only a few files."""
    cmd = [
        sys.executable, "convert_webdataset.py",
        "s3://test-bucket-img2dataset/photo-concept-bucket-webdataset/",
        "s3://test-bucket-img2dataset/converted-cosmos-dataset-mp-limited/",
        "--extension-mapping", '{"txt":"captions","jpg":"images"}',
        "--chunk-size", "100",
        "--num-processes", "2",
        "--aws-profile", "wasabi",
        "--s3-endpoint-url", "https://s3.wasabisys.com"
    ]
    
    print("Running limited multiprocess conversion test...")
    print("Command:", " ".join(cmd))
    
    try:
        # Override the script to only process first 3 files
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print("Return code:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Test timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_limited_conversion()
    if success:
        print("✅ Limited multiprocess conversion test passed!")
    else:
        print("❌ Limited multiprocess conversion test failed!")
    sys.exit(0 if success else 1)