"""
Caffe network visualization: draw the NetParameter protobuffer.

NOTE: this requires pydot>=1.0.2, which is not included in requirements.txt
since it requires graphviz and other prerequisites outside the scope of the
Caffe.
"""

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import pydot

# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record', 'fillcolor': '#6495ED',
         'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record', 'fillcolor': '#90EE90',
         'style': 'filled'}
BLOB_STYLE = {'shape': 'octagon', 'fillcolor': '#E0E0E0',
        'style': 'filled'}

def get_pooling_types_dict():
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe_pb2.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k,v in desc.values_by_name.items():
        d[v.number] = k
    return d


def determine_edge_label_by_layertype(layer, layertype):
    """Define edge label based on layer type
    """

    if layertype == 'Data':
        edge_label = 'Batch ' + str(layer.data_param.batch_size)
    elif layertype == 'Convolution':
        edge_label = str(layer.convolution_param.num_output)
    elif layertype == 'InnerProduct':
        edge_label = str(layer.inner_product_param.num_output)
    else:
        edge_label = '""'

    return edge_label


def determine_node_label_by_layertype(layer, layertype, rankdir):
    """Define node label based on layer type
    """

    if rankdir in ('TB', 'BT'):
        # If graph orientation is vertical, horizontal space is free and
        # vertical space is not; separate words with spaces
        separator = ' '
    else:
        # If graph orientation is horizontal, vertical space is free and
        # horizontal space is not; separate words with newlines
        separator = '\n'

    if layertype == 'Convolution':
        # Outer double quotes needed or else colon characters don't parse
        # properly
        node_label = '"%s%s(%s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      layertype,
                      separator,
                      layer.convolution_param.kernel_size,
                      separator,
                      layer.convolution_param.stride,
                      separator,
                      layer.convolution_param.pad)
    elif layertype == 'Pooling':
        pooling_types_dict = get_pooling_types_dict()
        node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      pooling_types_dict[layer.pooling_param.pool],
                      layertype,
                      separator,
                      la