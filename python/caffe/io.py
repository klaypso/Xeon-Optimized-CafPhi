import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize

try:
    # Python3 will most likely not be able to load protobuf
    from caffe.proto import caffe_pb2
except:
    import sys
    if sys.version_info >= (3,0):
        print("Failed to include caffe_pb2, things might go wrong!")
    else:
        raise

## proto / datum / ndarray conversion

def blobproto_to_array(blob, return_diff=False):
    """Convert a blob proto to an array. In default, we will just return the data,
    unless return_diff is True, in which case we will return the diff.
    """
    if return_diff:
        return np.array(blob.diff).reshape(
            blob.num, blo