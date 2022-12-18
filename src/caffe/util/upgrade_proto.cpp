#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <map>
#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

bool NetNeedsUpgrade(const NetParameter& net_param) {
  return NetNeedsV0ToV1Upgrade(net_param) || NetNeedsV1ToV2Upgrade(net_param);
}

bool NetNeedsV0ToV1Upgrade(const NetParameter& net_param) {
  for (int i = 0; i < net_param.layers_size(); ++i) {
    if (net_param.layers(i).has_layer()) {
      return true;
    }
  }
  return false;
}

bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param) {
  return net_param.layers_size() > 0;
}

bool UpgradeV0Net(const NetParameter& v0_net_param_padding_layers,
                  NetParameter* net_param) {
  // First upgrade padding layers to padded conv layers.
  NetParameter v0_net_param;
  UpgradeV0PaddingLayers(v0_net_param_padding_layers, &v0_net_param);
  // Now upgrade layer parameters.
  bool is_fully_compatible = true;
  net_param->Clear();
  if (v0_net_param.has_name()) {
    net_param->set_name(v0_net_param.name());
  }
  for (int i = 0; i < v0_net_param.layers_size(); ++i) {
    is_fully_compatible &= UpgradeV0LayerParameter(v0_net_param.layers(i),
                                                   net_param->add_layers());
  }
  for (int i = 0; i < v0_net_param.input_size(); ++i) {
    net_param->add_input(v0_net_param.input(i));
  }
  for (int i = 0; i < v0_net_param.input_dim_size(); ++i) {
    net_param->add_input_dim(v0_net_param.input_dim(i));
  }
  if (v0_net_param.has_force_backward()) {
    net_param->set_force_backward(v0_net_param.force_backward());
  }
  return is_fully_compatible;
}

void UpgradeV0PaddingLayers(const NetParameter& param,
                            NetParameter* param_upgraded_pad) {
  // Copy everything other than the layers from the original param.
  param_upgraded_pad->Clear();
  param_upgraded_pad->CopyFrom(param);
  param_upgraded_pad->clear_layers();
  // Figure out which layer each bottom blob comes from.
  map<string, int> blob_name_to_last_top_idx;
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    blob_name_to_last_top_idx[blob_name] = -1;
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    const V1LayerParameter& layer_connection = param.layers(i);
    const V0LayerParameter& layer_param = layer_connection.layer();
    // Add the layer to the new net, unless it's a padding layer.
    if (layer_param.type() != "padding") {
      param_upgraded_pad->add_layers()->CopyFrom(layer_connection);
    }
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const string& blob_name = layer_connection.bottom(j);
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown blob input " << blob_name << " to layer " << j;
      }
      const int top_idx = blob_name_to_last_top_idx[blob_name];
      if (top_idx == -1) {
        continue;
      }
      const V1LayerParameter& source_layer = param.layers(top_idx);
      if (source_layer.layer().type() == "padding") {
        // This layer has a padding layer as input -- check that it is a conv
        // layer or a pooling layer and takes only one input.  Also check that
        // the padding layer input has only one input and one output.  Other
        // cases have undefined behavior in Caffe.
        CHECK((layer_param.type() == "conv") || (layer_param.type() == "pool"))
            << "Padding layer input to "
            "non-convolutional / non-pooling layer type "
            << layer_param.type();
        CHECK_EQ(layer_connection.bottom_size(), 1)
            << "Conv Layer takes a single blob as input.";
        CHECK_EQ(source_layer.bottom_size(), 1)
            << "Padding Layer takes a single blob as input.";
        CHECK_EQ(source_layer.top_size(), 1)
            << "Padding Layer produces a single blob as output.";
        int layer_index = param_upgraded_pad->layers_size() - 1;
        param_upgraded_pad->mutable_layers(layer_index)->mutable_layer()
            ->set_pad(source_layer.layer().pad());
        param_upgraded_pad->mutable_layers(layer_index)
            ->set_bottom(j, s