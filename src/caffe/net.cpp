#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/CycleTimer.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  param.mutable_state()->set_phase(phase);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  LOG(INFO) << "Initializing net from parameters: " << std::endl
            << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  CHECK(param.input_dim_size() == 0 || param.input_shape_size() == 0)
      << "Must specify either input_shape OR deprecated input_dim, not both.";
  if (param.input_dim_size() > 0) {
    // Deprecated 4D dimensions.
    CHECK_EQ(param.input_size() * 4, param.input_dim_size())
        << "Incorrect input blob dimension specifications.";
  } else {
    CHECK_EQ(param.input_size(), param.input_shape_size())
        << "Exactly one input_shape must be specified per input.";
  }
  memory_used_ = 0;
  // set the input blobs
  for (int input_id = 0; input_id < param.input_size(); ++input_id) {
    const int layer_id = -1;  // inputs have fake layer ID -1
    AppendTop(param, layer_id, input_id, &available_blobs, &blob_name_to_idx);
  }
  DLOG(INFO) << "XEON: Memory required for data: " << memory_used_ * sizeof(Dtype);
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.lay